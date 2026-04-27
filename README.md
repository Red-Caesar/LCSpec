# The Study of Speculative Decoding on a Long Context

## Analysis

Analysis can be found at `analysis/results.ipynb`

## Setup

1. **Clone and Update Submodules**
```bash
git clone https://github.com/Red-Caesar/LCSpec.git
cd LCSpec
git submodule update --init --recursive
```

2. **Create Virtual Environment** (using uv)
```bash
make install
source .venv/bin/activate
```

## Collecting Data

All experiment scripts live in the `experiments/` folder. Run them from the project root with the virtual environment active.

### Speculative Decoding (SD) tests

Measures acceptance rates for SD setups across a range of input token lengths.
```bash
bash experiments/run_sd.sh
```
Results are written to `results/sd_experiments/`.

### Load tests

Sends concurrent requests to a running vLLM server and records latency under load.
```bash
bash experiments/run_load_test.sh
```
Artifacts land in `perfomance/scripts/load_test/artifacts/`.

### Decode statistics

Plot histograms showing how many decodes are made per request..

```bash
bash experiments/run_decode_statistics.sh
```
Results are written to `results/decode_statistics/`.


## Loading Results into the Database

After collecting results, load them into the database with `experiments/load_db.sh`.

```bash
bash experiments/load_db.sh all
```

### Local SQLite
By default the script writes to `storage/lcspec.db`. No extra configuration needed.

### Cloud DB (Supabase)
To push results to the shared Supabase PostgreSQL instance, set `SUPABASE_DB_URL` in the `.env` file at the project root:

```
SUPABASE_DB_URL=postgresql://<user>:<password>@<host>:<port>/postgres
```
