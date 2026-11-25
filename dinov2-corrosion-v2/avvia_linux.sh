#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export HF_HOME="$PWD/model_cache"
export DINO_CPU_THREADS=4
.venv/bin/python main.py
