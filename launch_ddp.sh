#!/usr/bin/env bash
set -e
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train.py --backbone PARENet --dataset P2ILReg --method Source_Only "$@"