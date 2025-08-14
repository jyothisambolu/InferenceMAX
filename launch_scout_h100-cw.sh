#!/usr/bin/env bash

set -euo pipefail
set -x

GHA_CACHE_DIR="/mnt/vast/"
digits="${USER//[^0-9]/}"
export PORT_OFFSET=$(( ${digits:0:1} * (${digits: -1} + 1) ))

JOB_SCRIPT=$(mktemp $GITHUB_WORKSPACE/slurm-XXXXXX.sh)
cat > $JOB_SCRIPT << 'EOF'
#!/usr/bin/env bash

echo "JOB \$SLURM_JOB_ID running on NODE \$SLURMD_NODENAME"

port=$(( 8888 + $PORT_OFFSET ))

set -x
vllm serve $MODEL --port $port \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype auto --max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN \
--disable-log-requests > /results/server_${SLURM_JOB_ID}.log 2>&1 &

set +x
while ! grep -q "Application startup complete\." /results/server_${SLURM_JOB_ID}.log; do
    if grep -i "error" /results/server_${SLURM_JOB_ID}.log; then
        grep -iC5 "error" /results/server_${SLURM_JOB_ID}.log
        exit
    fi
    tail -n10 /results/server_${SLURM_JOB_ID}.log
    sleep 5
done
tail -n10 /results/server_${SLURM_JOB_ID}.log

git clone -b v0.7.3 https://github.com/vllm-project/vllm.git
set -x
python3 vllm/benchmarks/benchmark_serving.py \
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
chmod u+x $JOB_SCRIPT

srun --partition=h100 --gres=gpu:$TP \
--container-image=$IMAGE \
--container-mounts=$GHA_CACHE_DIR:/mnt/,$GITHUB_WORKSPACE:/results/ \
--no-container-entrypoint \
--export=ALL \
bash < $JOB_SCRIPT
