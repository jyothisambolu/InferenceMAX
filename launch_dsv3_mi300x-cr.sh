#!/usr/bin/env bash

GHA_CACHE_DIR="/mnt/vdb/gha_cache/"

network_name="bmk-net"
server_name="bmk-server-$RANDOM"
client_name="bmk-client"
port=8888

docker network create $network_name

docker ps -q | xargs -r docker rm -f
while [ -n "$(docker ps -aq)" ]; do
  sleep 5
done

set -x
docker run --rm -d --ipc host --network $network_name --name $server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $GHA_CACHE_DIR:/mnt/ -e HF_TOKEN=$HF_TOKEN -e HF_HUB_CACHE=$HF_HUB_CACHE -e SGLANG_AITER_MOE=1 \
$IMAGE \
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $port --trust-remote-code --tp $TP --cuda-graph-max-bs $CONC

set +x
while ! docker logs $server_name 2>&1 | grep -q "The server is fired up and ready to roll!"; do
    docker logs --tail 10 $server_name
    sleep 5
done

docker run --rm --network $network_name --name $client_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $GITHUB_WORKSPACE:/results/ -e HF_TOKEN=$HF_TOKEN \
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
--result-dir /results/ --result-filename $RESULT_FILENAME.json"

docker stop $server_name $client_name
docker network rm $network_name
