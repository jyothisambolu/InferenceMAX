#!/usr/bin/env bash

sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

while [ -n "$(docker ps -aq)" ]; do
  sleep 5
done

GHA_CACHE_DIR="/mnt/vdb/gha_cache/"

network_name="bmk-net"
server_name="bmk-server-$RANDOM"
client_name="bmk-client"
port=8888

docker network create $network_name

set -x
docker run --rm -d --ipc host --shm-size=16g --network $network_name --name $server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $GHA_CACHE_DIR:/mnt/ -e HF_TOKEN=$HF_TOKEN -e HF_HUB_CACHE=$HF_HUB_CACHE -e VLLM_USE_TRITON_FLASH_ATTN=0 $IMAGE \
vllm serve $MODEL --port $port \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype bfloat16 --quantization fp8 \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN \
--no-enable-prefixcaching \
--disable-log-requests

set +x
while ! docker logs $server_name 2>&1 | grep -Fq "Application startup complete."; do
    if docker logs $server_name 2>&1 | grep -iq "error"; then
        docker logs $server_name 2>&1 | grep -iC5 "error"
        exit 1
    fi
    docker logs --tail 10 $server_name
    sleep 5
done
docker logs --tail 10 $server_name

git clone https://github.com/kimbochen/bench_serving.git 

set -x
docker run --rm --network $network_name --name $client_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ -e HF_TOKEN=$HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=/bin/bash $IMAGE -c \
"python3 bench_serving/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://$server_name:$port \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json"

docker stop $server_name
docker network rm $network_name
