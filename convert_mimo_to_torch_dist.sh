#!/bin/bash
# Converts MiMo-7B-RL HF checkpoint to Megatron torch_dist format
# Needed for --ref-load argument in training

set -ex

MILES_PYTHON=/home/rs63759/micromamba/envs/miles/bin/python3.12
SCRIPT=/home/rs63759/code/spec/Trajectory-Conditioned-Speculative-Decoding/tools/convert_hf_to_torch_dist.py
MEGATRON=/home/rs63759/code/spec/Megatron-LM
HF_CKPT=/home/rs63759/code/spec/models/MiMo-7B-RL
SAVE_DIR=/home/rs63759/code/spec/models/MiMo-7B-RL_torch_dist

mkdir -p ${SAVE_DIR}

PYTHONPATH=${MEGATRON} \
/home/rs63759/micromamba/envs/miles/bin/torchrun --nproc_per_node=1 ${SCRIPT} \
    --hf-checkpoint ${HF_CKPT} \
    --save ${SAVE_DIR} \
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
