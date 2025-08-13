#!/usr/bin/env bash

GHA_CACHE_DIR="/raid/"
PORT_OFFSET=${USER: -1}

JOB_SCRIPT=$(mktemp $PWD/slurm-XXXXXX.sh)
cat > $JOB_SCRIPT <<-EOF
#!/usr/bin/env bash

huggingface-cli download $MODEL

set -x
PORT=$(( 8888 + $PORT_OFFSET ))
export SGL_ENABLE_JIT_DEEPGEMM=0
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port \$PORT --trust-remote-code \
--tp $TP --dp 1 \
--max-running-requests $CONC --cuda-graph-max-bs $CONC \
--disable-radix-cache --chunked-prefill-size 32768 --mem-fraction-static 0.89 --max-prefill-tokens 32768 \
--attention-backend trtllm_mla --disable-shared-experts-fusion --enable-flashinfer-trtllm-moe \
> /workspace/server_\${SLURM_JOB_ID}.log 2>&1 &

set +x
while ! grep -q "The server is fired up and ready to roll!" /workspace/server_\${SLURM_JOB_ID}.log; do
    if grep -iq "error" /workspace/server_\${SLURM_JOB_ID}.log; then
        grep -iC5 "error" /workspace/server_\${SLURM_JOB_ID}.log
        exit 1
    fi
    tail -n10 /workspace/server_\${SLURM_JOB_ID}.log
    sleep 5
done
tail -n10 /workspace/server_\${SLURM_JOB_ID}.log

git clone -b v0.7.3 https://github.com/vllm-project/vllm.git
set -x
python3 vllm/benchmarks/benchmark_serving.py \
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
--no-container-entrypoint \
--export=ALL \
bash < $JOB_SCRIPT

rm -f $JOB_SCRIPT
