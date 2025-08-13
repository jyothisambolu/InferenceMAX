#!/usr/bin/bash

GHA_CACHE_DIR="/raid/"
PORT_OFFSET=${USER: -1}

JOB_SCRIPT=$(mktemp $GITHUB_WORKSPACE/slurm-XXXXXX.sh)
cat > $JOB_SCRIPT <<-EOF
#!/usr/bin/env bash

hf download $MODEL
PORT=$(( 8888 + $PORT_OFFSET ))
set -x
vllm serve $MODEL --port \$PORT \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype bfloat16 --quantization modelopt \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN \
--disable-log-requests > /workspace/server_\${SLURM_JOB_ID}.log 2>&1 &

set +x
while ! grep -q "Application startup complete\." /workspace/server_\${SLURM_JOB_ID}.log; do
    if grep -iq "error" /workspace/server_\${SLURM_JOB_ID}.log; then
        grep -iC5 "error" /workspace/server_\${SLURM_JOB_ID}.log
        exit 1
    fi
    tail -n10 /workspace/server_\${SLURM_JOB_ID}.log
    sleep 5
done
tail -n10 /workspace/server_\${SLURM_JOB_ID}.log

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
srun --partition=dgx-h200 --gres=gpu:$TP --exclusive \
--container-image=$IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$GHA_CACHE_DIR/hf_hub_cache/:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash < $JOB_SCRIPT

rm -f $JOB_SCRIPT
