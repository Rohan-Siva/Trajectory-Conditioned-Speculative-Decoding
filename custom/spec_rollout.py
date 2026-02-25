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
