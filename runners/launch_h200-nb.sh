#!/usr/bin/bash

MODEL_CODE="${1%%_*}"
export HF_HUB_CACHE_MOUNT="/home/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

PARTITION="main"
# Use framework-specific SQSH file
if [ "$FRAMEWORK" = "trt" ]; then
    SQUASH_FILE="/home/squash/image_${MODEL_CODE}_h200_trt.sqsh"
else
    SQUASH_FILE="/home/squash/image_${MODEL_CODE}_h200.sqsh"
fi

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

set -x
srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$(realpath $SQUASH_FILE) \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash -c "
# Determine which benchmark script to use based on framework
if [ \"\$FRAMEWORK\" = \"trt\" ]; then
    bash benchmarks/\${MODEL_CODE}_h200_trt_slurm.sh
else
    bash benchmarks/\${MODEL_CODE}_h200_slurm.sh
fi
"

scancel $JOB_ID
