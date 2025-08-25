#!/usr/bin/bash

MODEL_CODE="${1%%_*}"
HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
source benchmarks/${MODEL_CODE}_h100_docker.sh
