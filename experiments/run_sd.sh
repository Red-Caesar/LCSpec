#!/bin/bash

CONFIGS=(
    # "perfomance/configs/acceptance_test/baseline.yaml"
    "perfomance/configs/acceptance_test/trained_spec.yaml"
    # "perfomance/configs/acceptance_test/partial.yaml"
    # "perfomance/configs/acceptance_test/dynamic_yarn.yaml"
)

DATASET="summary"           # code | summary | chat
SETUP_TYPE="few"            # single | few
NUM_PROMPTS=200              # -1 for all
OUTPUT_TOKENS=256
OUTPUT_DIR="results/sd_experiments"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

SMALL_INPUT_TOKENS="500:2001:500"
for CONFIG in "${CONFIGS[@]}"; do
    echo "Running SD: $CONFIG"
    python -m perfomance.scripts.run_sd \
        --config "$ROOT_DIR/$CONFIG" \
        --dataset "$DATASET" \
        --setup-type "$SETUP_TYPE" \
        --num-prompts "$NUM_PROMPTS" \
        --output-tokens "$OUTPUT_TOKENS" \
        --output-dir "$ROOT_DIR/$OUTPUT_DIR" \
        ${SMALL_INPUT_TOKENS:+--input-tokens "$SMALL_INPUT_TOKENS"}
done

LARGE_INPUT_TOKENS="4000:8001:4000"
for CONFIG in "${CONFIGS[@]}"; do
    echo "Running SD: $CONFIG"
    python -m perfomance.scripts.run_sd \
        --config "$ROOT_DIR/$CONFIG" \
        --dataset "$DATASET" \
        --setup-type "$SETUP_TYPE" \
        --num-prompts "$NUM_PROMPTS" \
        --output-tokens "$OUTPUT_TOKENS" \
        --output-dir "$ROOT_DIR/$OUTPUT_DIR" \
        ${LARGE_INPUT_TOKENS:+--input-tokens "$LARGE_INPUT_TOKENS"}
done
