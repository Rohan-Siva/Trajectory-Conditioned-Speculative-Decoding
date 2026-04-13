
#!/bin/bash

# Patched version of run-mimo-7B-rl-eagle.sh for local 4xA6000 setup
# Model paths: /home/ubuntu/code/models/
# Data paths:  /home/ubuntu/code/data/

# Full paths so this works in screen/tmux without conda activation
RAY=/home/ubuntu/micromamba/envs/miles/bin/ray
PYTHON=/home/ubuntu/micromamba/envs/miles/bin/python3.12
export PATH=/home/ubuntu/micromamba/envs/miles/bin:$PATH

export RAY_TMPDIR=/opt/dlami/nvme/ray
mkdir -p $RAY_TMPDIR

# for rerun the task
pkill -9 sglang
sleep 3
$RAY stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/home/ubuntu/code/Trajectory-Conditioned-Speculative-Decoding/scripts/models/mimo-7B-rl.sh"

BASE=/home/ubuntu/code

CKPT_ARGS=(
   --hf-checkpoint ${BASE}/models/MiMo-7B-RL
   --ref-load ${BASE}/models/MiMo-7B-RL_torch_dist
   --load ${BASE}/checkpoints/MiMo-7B-RL-mtp_miles/
   --save ${BASE}/checkpoints/MiMo-7B-RL-mtp_miles/
   --save-interval 2000
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE}/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${BASE}/data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project spec
   --wandb-group mimo-7B-rl-test
   --wandb-key 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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
   --sglang-watchdog-timeout 3600

   # for speculative decoding
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4

   # sometimes flashinfer has IMA bugs. Use fa3 instead
   --sglang-attention-backend fa3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

SPEC_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)

# launch the master node of ray — 4 GPUs (A6000)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_memory_usage_threshold=0.99
$RAY start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/home/ubuntu/code/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"RAY_memory_usage_threshold\": \"0.99\"
  }
}"

cd /home/ubuntu/code/Trajectory-Conditioned-Speculative-Decoding

$RAY job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${SPEC_ARGS[@]}
