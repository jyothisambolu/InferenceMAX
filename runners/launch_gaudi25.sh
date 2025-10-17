#!/usr/bin/bash

#HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
HF_HUB_CACHE_MOUNT="/home/jsambolu/hf_hub_cache/"
PORT=8888

server_name="bmk-server"
client_name="bmk-client"

set -x
docker run --rm -d --network=host --name=$server_name \
--runtime=habana --cap-add=sys_nice -v /software/data/pytorch/huggingface/hub:/root/.cache/huggingface/hub --privileged --ipc=host --shm-size=16g \
-v /software:/software \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/work/ -w /work/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e PORT=$PORT \
-e HABANA_VISIBLE_DEVICES=all \
--entrypoint=/bin/bash \
$IMAGE \
/work/benchmarks/"${EXP_NAME%%_*}_${PRECISION}_gaudi3_vm_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

if ! docker ps --format "{{.Names}}" | grep -q "$server_name"; then
    echo "Server container launch failed."
    exit 1
fi

git clone https://github.com/kimbochen/bench_serving.git

set -x
docker run --rm --network=host --name=$client_name \
-v $GITHUB_WORKSPACE:/work/ -w /work/ \
-e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=/bin/bash \
$IMAGE \
-lc "pip install -q datasets pandas && \
python3 bench_serving/benchmark_serving.py \
--model=$MODEL \
--backend=vllm \
--base-url=\"http://localhost:$PORT\" \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$(( $CONC * 10 )) --max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics='ttft,tpot,itl,e2el' \
--result-dir=/work/ \
--result-filename=$RESULT_FILENAME.json"

docker stop $server_name
