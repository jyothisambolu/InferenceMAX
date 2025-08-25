#!/usr/bin/bash

IMAGE_SQUASH="/home/.tw/slinky/.cache/squash/lmsysorg+sglang+v0.4.9.post2-rocm630-mi30x.sqsh"
HF_HUB_CACHE_MOUNT="/home/hf_hub_cache/"
PORT_OFFSET=${USER: -1}

JOB_SCRIPT=$(mktemp $GITHUB_WORKSPACE/slurm-XXXXXX.sh)
cat > $JOB_SCRIPT <<-EOF
#!/usr/bin/env bash

echo "JOB \$SLURM_JOB_ID running on \$SLURMD_NODENAME"

SERVER_LOG=\$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))
huggingface-cli download $MODEL

set -x
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port \$PORT --trust-remote-code \
--tp $TP --cuda-graph-max-bs $CONC --disable-radix-cache \
> \$SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "\$line"
    if [[ "\$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        exit 1
    fi
    if [[ "\$line" == *"The server is fired up and ready to roll"* ]]; then
        sleep 5
        tail -n100 "\$SERVER_LOG"
        echo "JOB \$SLURM_JOB_ID ran on \$SLURMD_NODENAME"
        break
    fi
done < <(tail -F -n0 "\$SERVER_LOG")

git clone https://github.com/kimbochen/bench_serving.git
set -x
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:\$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ \
--result-filename $RESULT_FILENAME.json
EOF

set -x
srun --partition=gpuworker --gres=gpu:$TP -c 128 \
--container-image=$IMAGE_SQUASH \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash < $JOB_SCRIPT

rm -f $JOB_SCRIPT
