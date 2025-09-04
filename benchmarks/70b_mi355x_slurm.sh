#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# PORT
# RESULT_FILENAME

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
vllm serve $MODEL --port $PORT \
--tensor-parallel-size $TP --distributed-executor-backend mp --dtype auto \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN \
--no-enable-prefix-caching \
--disable-log-requests > $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        sleep 3
        tail -n100 "$SERVER_LOG"
        exit 1
    fi
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url "http://0.0.0.0:$PORT" \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics "ttft,tpot,itl,e2el" \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json

exit
