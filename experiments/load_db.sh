#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

SD_DIR="$ROOT_DIR/results/sd_experiments"
LOAD_TEST_DIR="$ROOT_DIR/perfomance/scripts/load_test/artifacts"
DB_NAME="lcspec.db"

usage() {
    echo "Usage: $0 <load_test|sd_test|all> [--db-name <name>]"
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage
fi

run_sd() {
    echo "Loading SD experiment results from $SD_DIR"
    python -m perfomance.database.run \
        --etl_class sd_metrics \
        --data_dir "$SD_DIR" \
        --db_name "$DB_NAME"
}

run_load_test() {
    echo "Loading load test results from $LOAD_TEST_DIR"
    python -m perfomance.database.run \
        --etl_class load_test_metrics \
        --data_dir "$LOAD_TEST_DIR" \
        --db_name "$DB_NAME"
}

TYPE="$1"
case "$TYPE" in
    sd_test)
        run_sd
        ;;
    load_test)
        run_load_test
        ;;
    all)
        run_sd
        run_load_test
        ;;
esac
