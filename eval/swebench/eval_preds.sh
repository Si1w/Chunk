#!/bin/bash

cd /home/steven/home/chunk
set -e

DATASET="SWE-bench/SWE-bench_Lite"
MODEL="devstral-small-latest"
SPLIT="dev"

METHODS=("sliding" "function" "hierarchical" "cAST" "natural")
for METHOD in "${METHODS[@]}"; do
    PREDICTIONS_DIR="./eval/swebench/predictions/devstral/Qwen3-Embedding-0.6B/${METHOD}_predictions.json"
    python -m swebench.harness.run_evaluation \
        -d $DATASET \
        -s $SPLIT \
        -p $PREDICTIONS_DIR \
        -id "${METHOD}_${SPLIT}"
done