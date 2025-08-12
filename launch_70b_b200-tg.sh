#!/usr/bin/env bash

GHA_CACHE_DIR=/dev/shm/gha_cache/

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"
port=8888

docker network create $network_name

set -x
docker run --rm -d --network $network_name --name $server_name \
--runtime nvidia --gpus all --ipc host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $GHA_CACHE_DIR:/mnt/ -v $HOME/ComputeCache:/root/.nv/ComputeCache \
-e HF_TOKEN=$HF_TOKEN -e HF_HUB_CACHE=$HF_HUB_CACHE $IMAGE \
--model $MODEL --port $port \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype bfloat16 --quantization modelopt \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN \
--disable-log-requests

set +x
while ! docker logs $server_name 2>&1 | grep -q "Application startup complete\."; do
    if docker logs $server_name 2>&1 | grep -iq "error"; then
        docker logs $server_name 2>&1 | grep -iC5 "error"
        exit 1
    fi
    docker logs --tail 10 $server_name
    sleep 5
done

set -x
docker run --rm --network $network_name --name $client_name \
--runtime nvidia \
-v $GITHUB_WORKSPACE:/results/ -e HF_TOKEN=$HF_TOKEN \
--entrypoint /bin/bash $IMAGE -c \
"python3 benchmarks/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://$server_name:$port \
--dataset-name random \
--random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.2 \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /results/ --result-filename $RESULT_FILENAME.json"

docker stop $server_name $client_name
docker network rm $network_name
