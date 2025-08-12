#!/usr/bin/env bash

GHA_CACHE_DIR=/home/ubuntu/gha_cache/

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"
port=8888

docker network create $network_name

docker ps -q | xargs -r docker rm -f
while [ -n "$(docker ps -aq)" ]; do
  sleep 5
done

set -x
docker run --rm -d --network $network_name --name $server_name \
--runtime nvidia --gpus all --ipc host --shm-size=16g --privileged --ulimit memlock=-1 --ulimit stack=67108864 \
-v $GHA_CACHE_DIR:/mnt/ -e HF_TOKEN=$HF_TOKEN -e HF_HUB_CACHE=$HF_HUB_CACHE \
--entrypoint=/bin/bash $IMAGE -c \
"vllm serve $MODEL --port $port \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype bfloat16 --quantization modelopt \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN \
--disable-log-requests"

set +x
while ! docker logs $server_name 2>&1 | grep -Fq "Application startup complete."; do
    docker logs --tail 10 $server_name
    sleep 5
done
docker logs --tail 10 $server_name

docker run --rm --network $network_name --name $client_name \
--runtime nvidia \
-v $GITHUB_WORKSPACE:/workspace/results/ -w /workspace/ -e HF_TOKEN=$HF_TOKEN \
--entrypoint=/bin/bash $IMAGE -c \
"git clone -b v0.7.3 https://github.com/vllm-project/vllm.git && \
python3 vllm/benchmarks/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://$server_name:$port \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/results/ --result-filename $RESULT_FILENAME.json"

docker stop $server_name
docker network rm $network_name
