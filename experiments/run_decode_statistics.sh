#!/bin/bash

CONFIGS=(
    "perfomance/configs/decode_statistics/all.yaml"
)

MAX_TOKENS=30
NUM_PROMPTS=100
OUTPUT_DIR="results/decode_statistics"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

INPUT_TOKENS="1000:4001:3000"
for CONFIG in "${CONFIGS[@]}"; do
    echo "Running decode statistics: $CONFIG"
    python -m perfomance.scripts.run_decode_statistics \
        --config "$ROOT_DIR/$CONFIG" \
        --max-tokens "$MAX_TOKENS" \
        --num-prompts "$NUM_PROMPTS" \
        --input-tokens "$INPUT_TOKENS" \
        --output-dir "$ROOT_DIR/$OUTPUT_DIR"
done
