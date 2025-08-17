#!/usr/bin/env bash

set -euo pipefail
set -x

GHA_CACHE_DIR="/home/"
export PORT_OFFSET="${USER//[^0-9]/}"

JOB_SCRIPT=$(mktemp $GITHUB_WORKSPACE/slurm-XXXXXX.sh)
cat > $JOB_SCRIPT << 'EOF'
#!/usr/bin/env bash

port=$(( 8888 + $PORT_OFFSET ))

set -x
vllm serve $MODEL --host 0.0.0.0 --port $port \
--trust-remote-code --quantization modelopt --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}' \
--disable-log-requests > /results/server_${SLURM_JOB_ID}.log 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        echo "JOB $SLURM_JOB_ID ran on NODE $SLURMD_NODENAME"
        exit 1
    fi
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "/results/server_$SLURM_JOB_ID.log")

git clone https://github.com/kimbochen/bench_serving.git 
set -x
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:$port \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /results/ \
--result-filename $RESULT_FILENAME.json
EOF

srun --partition=main --gres=gpu:$TP \
--container-image=$IMAGE \
--container-mounts=$GHA_CACHE_DIR:/mnt/,$GITHUB_WORKSPACE:/results/ \
--no-container-entrypoint \
--export=ALL \
bash < $JOB_SCRIPT
