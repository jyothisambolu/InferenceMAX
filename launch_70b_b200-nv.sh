#!/usr/bin/bash

GHA_CACHE_DIR="/raid/"
PORT_OFFSET=${USER: -1}

JOB_SCRIPT=$(mktemp $GITHUB_WORKSPACE/slurm-XXXXXX.sh)
cat > $JOB_SCRIPT <<-EOF
#!/usr/bin/env bash

echo "JOB \$SLURM_JOB_ID running on NODE \$SLURMD_NODENAME"

SERVER_LOG=\$(mktemp /workspace/server-XXXXXX.log)

hf download $MODEL
PORT=$(( 8888 + $PORT_OFFSET ))
set -x
vllm serve $MODEL --host 0.0.0.0 --port \$PORT \
--trust-remote-code --quantization modelopt --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}' \
--disable-log-requests > \$SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "\$line"
    if [[ "\$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        echo "JOB \$SLURM_JOB_ID ran on NODE \$SLURMD_NODENAME"
        exit 1
    fi
    if [[ "\$line" == *"Application startup complete"* ]]; then
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
srun --partition=dgx-b200 --gres=gpu:$TP --exclusive \
--container-image=$IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$GHA_CACHE_DIR/hf_hub_cache/:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash < $JOB_SCRIPT

rm -f $JOB_SCRIPT
