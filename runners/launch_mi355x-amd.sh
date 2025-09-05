#!/usr/bin/env bash

# === Workflow-defined Env Vars ===
# IMAGE
# MODEL
# TP
# HF_HUB_CACHE
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# CONC
# GITHUB_WORKSPACE
# RESULT_FILENAME
# HF_TOKEN

MODEL_CODE="${1%%_*}"
HF_HUB_CACHE_MOUNT="/shared/amdgpu/home/kimbo_o80/hf_hub_cache/"
export PORT=8888

set -x
srun --reservation=PU74C0_reservation --exclusive \
--gres=gpu:$TP --cpus-per-task=128 --ntasks-per-node=1 --time=180 \
--container-image=$IMAGE \
--container-name="${MODEL_CODE}_container" \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${MODEL_CODE}_mi355x_slurm.sh
