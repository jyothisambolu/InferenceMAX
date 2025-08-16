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
docker run --rm -d --ipc host --shm-size=16g --network $network_name --name $server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HOME_DIR/hf_hub_cache/:$HF_HUB_CACHE \
-e HF_TOKEN=$HF_TOKEN -e HF_HUB_CACHE=$HF_HUB_CACHE -e VLLM_USE_TRITON_FLASH_ATTN=0 \
--entrypoint=python3 \
$IMAGE \
-m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $port --trust-remote-code \
--tp $TP --cuda-graph-max-bs $CONC

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        docker stop $server_name
        exit 1
    fi
    if [[ "$line" == *"The server is fired up and ready to roll!"* ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

git clone https://github.com/kimbochen/bench_serving.git

set -x
docker run --rm --network $network_name --name $client_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
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
