# Load Testing

This directory contains Locust-based load testing scripts for LLM inference endpoints.

## Quick Start

```bash
# Basic test with default settings
python load_test.py

# Custom configuration
python load_test.py \
  --host http://localhost:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --users 20 \
  --spawn-rate 2 \
  --run-time 60s
```

### With Web UI

```bash
python load_test.py --web-ui
```

## Configuration

### Command Line Arguments

The `load_test.py` script accepts the following arguments:

```
--host              Target server host URL
--model             Model name to test
--users, -u         Number of concurrent users
--spawn-rate, -r    Users spawned per second
--run-time, -t      Test duration (e.g., 60s, 5m, 1h)
--dataset           Dataset type identifier
--artifacts-dir     Directory to save artifacts
--run-id            Run identifier
--web-ui            Run with web UI
--web-port          Port for web UI
```

TODO: change input and output tokens-distribution?

### Artifacts

The scripts generate the following files in the artifacts directory:

```
artifacts/
└── {run_id}_{timestamp}/
    ├── config.json          # Test configuration
    └── summary.json         # Aggregated metrics
```
