#!/usr/bin/bash

MODEL="${1%%_*}"
HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
source benchmarks/${MODEL}_h100_docker.sh
