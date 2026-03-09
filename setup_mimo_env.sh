#!/bin/bash
# ==============================================================================
# MiMo-7B-RL Setup Automation Script
#
# This script automatically downloads the MiMo-7B-RL model, converts datasets
# (DAPO-Math-17k and AIME-2024) into the required JSONL format, performs the
# Megatron torch_dist checkpoint conversion, and applies critical patches to
# the `miles` environment to prevent OOM and import errors during training.
# ==============================================================================

set -e

# Configuration
BASE_DIR="${BASE_DIR:-$(pwd)}"
MILES_REPO="${BASE_DIR}/Trajectory-Conditioned-Speculative-Decoding"
MEGATRON_REPO="${BASE_DIR}/Megatron-LM"

# You can override these if not using an active conda environment:
PYTHON_EXEC="${PYTHON_EXEC:-python3}"
TORCHRUN_EXEC="${TORCHRUN_EXEC:-torchrun}"

echo "=== MiMo-7B-RL Setup Script ==="
echo "Base Directory: $BASE_DIR"

if [ ! -d "${MILES_REPO}" ]; then
    echo "ERROR: Could not find miles repository at ${MILES_REPO}"
    echo "Please run this script from the parent directory of Trajectory-Conditioned-Speculative-Decoding."
    exit 1
fi

# 1. Create directory structure
echo ""
echo "=== 1. Setting up directories ==="
mkdir -p "${BASE_DIR}/models/MiMo-7B-RL"
mkdir -p "${BASE_DIR}/models/MiMo-7B-RL_torch_dist"
mkdir -p "${BASE_DIR}/checkpoints/MiMo-7B-RL-mtp_miles"
mkdir -p "${BASE_DIR}/data/dapo-math-17k/data"
mkdir -p "${BASE_DIR}/data/aime-2024"
echo "Directories configured."

# 2. Download MiMo-7B-RL Checkpoint
echo ""
echo "=== 2. Downloading MiMo-7B-RL HuggingFace Checkpoint ==="
if [ -z "$(ls -A "${BASE_DIR}/models/MiMo-7B-RL")" ]; then
    huggingface-cli download XiaomiMiMo/MiMo-7B-RL --local-dir "${BASE_DIR}/models/MiMo-7B-RL"
else
    echo "Model already downloaded at ${BASE_DIR}/models/MiMo-7B-RL. Skipping."
fi

# 3. Download & Prepare Datasets
echo ""
echo "=== 3. Downloading Datasets ==="
if [ ! -f "${BASE_DIR}/data/dapo-math-17k/data/dapo-math-17k.parquet" ]; then
    echo "Downloading DAPO-Math-17k..."
    huggingface-cli download Skywork/DAPO-Math-17k --repo-type dataset --local-dir "${BASE_DIR}/data/dapo-math-17k"
fi

if [ ! -f "${BASE_DIR}/data/aime-2024/aime_2024_problems.parquet" ]; then
    echo "Downloading AIME 2024..."
    # Note: adjust the repo name if the original downloaded parquet comes from a different HF source
    huggingface-cli download opencompmath/AIME_2024 --repo-type dataset --local-dir "${BASE_DIR}/data/aime-2024" || echo "Warn: AIME download failed, assure the parquet is in data/aime-2024/aime_2024_problems.parquet"
fi

echo ""
echo "=== 4. Converting Parquet Datasets to JSONL ==="
cat << 'EOF' > "${BASE_DIR}/prepare_mimo_data.py"
import json
import pandas as pd
import sys
from pathlib import Path

base_dir = sys.argv[1]

DAPO_PARQUET = f"{base_dir}/data/dapo-math-17k/data/dapo-math-17k.parquet"
DAPO_JSONL   = f"{base_dir}/data/dapo-math-17k/dapo-math-17k.jsonl"

AIME_PARQUET = f"{base_dir}/data/aime-2024/aime_2024_problems.parquet"
AIME_JSONL   = f"{base_dir}/data/aime-2024/aime-2024.jsonl"

try:
    if not Path(DAPO_JSONL).exists() and Path(DAPO_PARQUET).exists():
        print(f"Converting {DAPO_PARQUET}...")
        df = pd.read_parquet(DAPO_PARQUET)
        with open(DAPO_JSONL, "w") as f:
            for _, row in df.iterrows():
                prompt = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
                label = row["reward_model"]["ground_truth"]
                f.write(json.dumps({"prompt": prompt, "label": label}) + "\n")
        print(f"Saved: {DAPO_JSONL}")
    
    if not Path(AIME_JSONL).exists() and Path(AIME_PARQUET).exists():
        print(f"Converting {AIME_PARQUET}...")
        df = pd.read_parquet(AIME_PARQUET)
        AIME_SYSTEM = (
            "Solve the following math competition problem step by step. "
            "The last line of your response should be of the form Answer: $Answer "
            "(without quotes) where $Answer is the answer to the problem.\n\n"
        )
        with open(AIME_JSONL, "w") as f:
            for _, row in df.iterrows():
                prompt = [{"role": "user", "content": AIME_SYSTEM + str(row["Problem"])}]
                label = str(row["Answer"])
                f.write(json.dumps({"prompt": prompt, "label": label}) + "\n")
        print(f"Saved: {AIME_JSONL}")
    print("Dataset conversion finished.")
except Exception as e:
    print(f"Error preparing datasets: {e}")
EOF
$PYTHON_EXEC "${BASE_DIR}/prepare_mimo_data.py" "${BASE_DIR}"

# 5. Apply Miles Source Code Patches
echo ""
echo "=== 5. Applying Miles Codebase Patches ==="
cat << 'EOF' > "${BASE_DIR}/apply_miles_patches.py"
import os
import sys

miles_dir = sys.argv[1]

# 1. Patch numpy version check in initialize.py
init_sh = os.path.join(miles_dir, "miles/backends/megatron_utils/initialize.py")
with open(init_sh, "r") as f:
    content = f.read()
if "assert np.__version__.startswith(\"1.\")" in content:
    content = content.replace("assert np.__version__.startswith(\"1.\")", "# assert np.__version__.startswith(\"1.\")")
    with open(init_sh, "w") as f:
        f.write(content)
    print("Patched initialize.py (numpy version check ignored)")

# 2. Patch sglang import in quantizer_mxfp8.py
quantizer = os.path.join(miles_dir, "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py")
with open(quantizer, "r") as f:
    content = f.read()
target_import = "from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize"
if target_import in content:
    replacement = "try:\n    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize\nexcept ImportError:\n    mxfp8_group_quantize = None"
    content = content.replace(target_import, replacement)
    with open(quantizer, "w") as f:
        f.write(content)
    print("Patched quantizer_mxfp8.py (made mxfp8_group_quantize import optional)")

# 3. Patch distributed optimizer in arguments.py
args_py = os.path.join(miles_dir, "miles/backends/megatron_utils/arguments.py")
with open(args_py, "r") as f:
    content = f.read()
if "args.use_distributed_optimizer = True" in content:
    replacement = "args.use_distributed_optimizer = False  # Patched to reduce CPU RAM OOM"
    content = content.replace("args.use_distributed_optimizer = True", replacement)
    with open(args_py, "w") as f:
        f.write(content)
    print("Patched arguments.py (disabled use_distributed_optimizer)")
EOF
$PYTHON_EXEC "${BASE_DIR}/apply_miles_patches.py" "${MILES_REPO}"

# 6. Convert HF checkpint to Megatron torch_dist
echo ""
echo "=== 6. Converting MiMo-7B-RL to Megatron torch_dist ==="
if [ ! -f "${BASE_DIR}/models/MiMo-7B-RL_torch_dist/latest_checkpointed_iteration.txt" ]; then
    export PYTHONPATH=${MEGATRON_REPO}
    $TORCHRUN_EXEC --nproc_per_node=1 \
        "${BASE_DIR}/Trajectory-Conditioned-Speculative-Decoding/tools/convert_hf_to_torch_dist.py" \
        --hf-checkpoint "${BASE_DIR}/models/MiMo-7B-RL" \
        --save "${BASE_DIR}/models/MiMo-7B-RL_torch_dist" \
        --swiglu \
        --num-layers 36 \
        --hidden-size 4096 \
        --ffn-hidden-size 11008 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 8 \
        --use-rotary-position-embeddings \
        --disable-bias-linear \
        --add-qkv-bias \
        --normalization RMSNorm \
        --norm-epsilon 1e-05 \
        --rotary-base 640000 \
        --vocab-size 151680 \
        --untie-embeddings-and-output-weights \
        --max-position-embeddings 32768 \
        --mtp-num-layers 1 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --no-load-optim \
        --no-load-rng \
        --bf16
    echo "Model successfully converted to torch_dist format."
else
    echo "Megatron torch_dist model already present at ${BASE_DIR}/models/MiMo-7B-RL_torch_dist. Skipping conversion."
fi

echo ""
echo "========================================================================="
echo "Setup is complete!"
echo "You can now run the patched SLURM/bash run-mimo-7B-rl-eagle-local.sh file."
echo "========================================================================="
