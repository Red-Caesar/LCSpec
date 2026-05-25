#!/bin/bash

CONFIGS=(
    "perfomance/configs/load_test/baseline.yaml"
    "perfomance/configs/load_test/finetuned_official.yaml"
    # "perfomance/configs/load_test/tree_attention.yaml"
    # "perfomance/configs/load_test/compressed_kv_cache.yaml"
    # "perfomance/configs/load_test/trained_spec.yaml"
    # "perfomance/configs/load_test/open_model_diff_nstokens.yaml"
)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

SMALL_INPUT_TOKENS="500:2001:500"  # e.g. "1000:8000:1000" for range, or empty for config default
for CONFIG in "${CONFIGS[@]}"; do
    echo "Running load test: $CONFIG"
    python -m perfomance.scripts.run_load_test \
        --config "$ROOT_DIR/$CONFIG" \
        ${SMALL_INPUT_TOKENS:+--input-tokens "$SMALL_INPUT_TOKENS"}
done

LARGE_INPUT_TOKENS="3000:8001:1000"
for CONFIG in "${CONFIGS[@]}"; do
    echo "Running load test: $CONFIG"
    python -m perfomance.scripts.run_load_test \
        --config "$ROOT_DIR/$CONFIG" \
        ${LARGE_INPUT_TOKENS:+--input-tokens "$LARGE_INPUT_TOKENS"}
done
