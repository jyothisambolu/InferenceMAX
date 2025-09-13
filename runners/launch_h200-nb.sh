#!/usr/bin/bash

MODEL_CODE="${EXP_NAME%%_*}"
export HF_HUB_CACHE_MOUNT="/home/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

# Reset framework if vllm or sglang to use default script
if [ "$FRAMEWORK" = "vllm" ] || [ "$FRAMEWORK" = "sglang" ]; then
    FRAMEWORK=""
fi

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
bash benchmarks/${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK:+_$FRAMEWORK}_slurm.sh

scancel $JOB_ID
