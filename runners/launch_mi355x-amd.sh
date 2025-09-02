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
SQUASH_FILE="/shared/amdgpu/home/kimbo_o80/squash/image_${MODEL_CODE}_mi355x.sqsh"
export PORT=8888

set -x
salloc --reservation=PU74C0_reservation --exclusive --gres=gpu:$TP --cpus-per-task=128 --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash ${MODEL_CODE}_mi355x_slurm.sh
scancel $JOB_ID
