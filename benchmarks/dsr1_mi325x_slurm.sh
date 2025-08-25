#!/usr/bin/bash

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

SERVER_LOG=\$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))
huggingface-cli download $MODEL

set -x
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tp $TP --cuda-graph-max-bs $CONC --disable-radix-cache \
> $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        exit 1
    fi
    if [[ "$line" == *"The server is fired up and ready to roll"* ]]; then
        sleep 5
        tail -n100 "$SERVER_LOG"
        echo "JOB $SLURM_JOB_ID ran on $SLURMD_NODENAME"
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git
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
