#!/usr/bin/bash

MODEL_CODE="${1%%_*}"
HF_HUB_CACHE_MOUNT="/home/kimbosemianalysis/hf_hub_cache/"
source benchmarks/${MODEL_CODE}_mi325x_docker.sh
