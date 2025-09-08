#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

MODEL_CODE="${1%%_*}"
# Reset framework if vllm or sglang to use default script
if [ "$FRAMEWORK" = "vllm" ] || [ "$FRAMEWORK" = "sglang" ]; then
    FRAMEWORK=""
fi
PARTITION="dgx-b200"
# Use framework-specific SQSH file
if [ "$FRAMEWORK" = "trt" ]; then
    SQUASH_FILE="/raid/image_${MODEL_CODE}_b200_trt-0907.sqsh"
else
    SQUASH_FILE="/raid/image_${MODEL_CODE}_b200-0907.sqsh"
fi

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

set -x
srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${MODEL_CODE}_b200${FRAMEWORK:+_$FRAMEWORK}_slurm.sh

scancel $JOB_ID
