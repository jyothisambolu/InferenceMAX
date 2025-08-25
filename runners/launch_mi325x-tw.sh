#!/usr/bin/env bash

MODEL="${1%%_*}"
export HF_HUB_CACHE_MOUNT="/home/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

PARTITION="gpuworker"
SQUASH_FILE="/home/.tw/slinky/.cache/squash/image_${MODEL}_mi325x.sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --no-shell
JOB_ID=$(squeue -u $USER -h -o %A)

set -x
srun --jobid=$JOB_ID --pty bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$(realpath $SQUASH_FILE) \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${MODEL}_mi325x_slurm.sh

scancel $JOB_ID
