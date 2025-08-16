#!/usr/bin/env bash

while [ -n "$(docker ps -aq)" ]; do
  sleep 1
done

HF_HOME_DIR="/dev/shm/"

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"
port=8888

docker network create $network_name

set -x
docker run --rm -d --network $network_name --name $server_name \
--runtime nvidia --gpus all --ipc host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HOME_DIR/hf_hub_cache/:$HF_HUB_CACHE \
-e HF_TOKEN=$HF_TOKEN -e HF_HUB_CACHE=$HF_HUB_CACHE -e TORCH_CUDA_ARCH_LIST="10.0" -e NCCL_P2P_DISABLE=1 \
--entrypoint=/bin/bash \
$IMAGE \
-lc "python3 -m pip uninstall nvidia-nccl-cu12 -y && python3 -m pip install --no-deps nvidia-nccl-cu12==2.26.2.post1 && \
vllm serve $MODEL --host 0.0.0.0 --port $port \
--trust-remote-code --quantization modelopt --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{\"pass_config\": {\"enable_fi_allreduce_fusion\": true}, \"custom_ops\": [\"+rms_norm\"], \"level\": 3}' \
--disable-log-requests"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        docker stop $server_name
        exit 1
    fi
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

git clone https://github.com/kimbochen/bench_serving.git

set -x
docker run --rm --network $network_name --name $client_name \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ -e HF_TOKEN=$HF_TOKEN \
--entrypoint=python3 \
$IMAGE \
bench_serving/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://$server_name:$port \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics "ttft,tpot,itl,e2el" \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json

docker stop $server_name
docker network rm $network_name
