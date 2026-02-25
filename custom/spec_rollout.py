"""
custom/spec_rollout.py
======================
A custom single-turn rollout function for miles GRPO training that adds:

  1. A configurable system prompt prefix injected before every request.
  2. Per-sample logging of speculative-decoding acceptance stats.
  3. A hook (``custom_prompt_fn``) that lets callers modify the prompt
     on-the-fly based on the current ``Sample`` object and training step.

Usage
-----
Pass this module to the training script with::

    --rollout-function-path custom.spec_rollout.generate

To inject a custom system prompt set the environment variable::

    CUSTOM_SYSTEM_PROMPT="Think step by step."

Or supply a ``custom_prompt_fn`` by subclassing / monkey-patching
``CUSTOM_PROMPT_FN`` at import time before the rollout workers start.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Callable

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable prompt hook
# ---------------------------------------------------------------------------

# Override this at import time to provide fully dynamic prompt logic.
# Signature: (sample: Sample, rollout_id: int) -> str | list[dict]
CUSTOM_PROMPT_FN: Callable[[Sample, int], str | list[dict]] | None = None


def _build_prompt(
    sample: Sample,
    rollout_id: int,
    system_prompt: str | None,
) -> str | list[dict]:
    """Return a (possibly modified) prompt for this sample.

    Priority order:
      1. ``CUSTOM_PROMPT_FN`` if set — full control.
      2. ``system_prompt`` string — prepended as a system message when the
         prompt is already in chat-dict format, or prepended as plain text.
      3. Original ``sample.prompt`` unchanged.
    """
    if CUSTOM_PROMPT_FN is not None:
        return CUSTOM_PROMPT_FN(sample, rollout_id)

    prompt = sample.prompt

    if system_prompt:
        if isinstance(prompt, list):
            # Chat-format: insert system message at the front if absent
            if not prompt or prompt[0].get("role") != "system":
                prompt = [{"role": "system", "content": system_prompt}] + prompt
        else:
            # Plain-string format: simple prefix
            prompt = f"{system_prompt}\n\n{prompt}"

    return prompt


# ---------------------------------------------------------------------------
# Speculative-decoding logging helper
# ---------------------------------------------------------------------------

def _log_spec_info(sample: Sample, rollout_id: int) -> None:
    """Log draft-acceptance metrics for a single completed sample.

    This fires only when speculative decoding is active (i.e. the sample's
    ``spec_info`` contains non-zero draft token counts).  The stats are:

    * ``spec_accept_rate``  — fraction of draft tokens accepted by the verifier.
    * ``spec_accept_length``  — average accepted tokens per verification step.

    These mirror the fields already tracked in ``Sample.SpecInfo`` so that
    users can spot drift without waiting for the aggregated rollout log.
    """
    info = sample.spec_info
    if info.spec_draft_token_num == 0:
        return  # speculative decoding not active for this sample

    logger.debug(
        "rollout=%d sample=%s | spec_accept_rate=%.3f spec_accept_length=%.2f"
        " drafted=%d accepted=%d verifications=%d",
        rollout_id,
        sample.index,
        info.spec_accept_rate,
        info.spec_accept_length,
        info.spec_draft_token_num,
        info.spec_accept_token_num,
        info.spec_verify_ct,
    )


# ---------------------------------------------------------------------------
# Main rollout entry point
# ---------------------------------------------------------------------------

async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Custom single-turn rollout with prompt injection + spec-decoding logging.

    Drop-in replacement for ``miles.rollout.generate_hub.single_turn.generate``.
    Pass to the training script via::

        --rollout-function-path custom.spec_rollout.generate
    """
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params

    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, (
        f"Unexpected sample status: {sample.status}"
    )

    # ------------------------------------------------------------------
    # 1. Inject custom prompt before tokenization
    # ------------------------------------------------------------------
    system_prompt = os.environ.get("CUSTOM_SYSTEM_PROMPT", "")
    original_prompt = sample.prompt
    sample.prompt = _build_prompt(sample, input.rollout_id, system_prompt or None)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)

    # Restore original prompt so the Sample stored in the buffer reflects
    # the raw data, not the modified one (prevents double-injection on retry).
    sample.prompt = original_prompt

    # ------------------------------------------------------------------
    # 2. Handle partial rollout resumption (same as single_turn.py)
    # ------------------------------------------------------------------
    if len(sample.response) > 0:
        input_ids = sample.tokens
        sampling_params = deepcopy(sampling_params)
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
        if sampling_params["max_new_tokens"] <= 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    # ------------------------------------------------------------------
    # 3. Build and send the generation request
    # ------------------------------------------------------------------
    payload, halt_status = compute_request_payload(
        args,
        input_ids=input_ids,
        sampling_params=sampling_params,
        multimodal_inputs=sample.multimodal_inputs,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    output = await post(url, payload)
    await update_sample_from_response(args, sample, payload=payload, output=output)

    # ------------------------------------------------------------------
    # 4. Log per-sample speculative-decoding acceptance stats
    # ------------------------------------------------------------------
    _log_spec_info(sample, input.rollout_id)

    return GenerateFnOutput(samples=sample)

