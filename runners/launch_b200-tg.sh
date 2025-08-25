#!/usr/bin/bash

MODEL="${1%%_*}"
HF_HUB_CACHE_MOUNT="/dev/shm/hf_hub_cache/"
source benchmarks/${MODEL}_b200_docker.sh
