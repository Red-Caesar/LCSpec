# The Study of Speculative Decoding on a Long Context

## Setup

1. **Clone and Update Submodules**
```bash
git clone https://github.com/Red-Caesar/LCSpec.git
cd LCSpec
git submodule update --init --recursive
```

2. **Create Virtual Environment** (using uv)
```bash
uv venv --python 3.12 --python-preference only-managed
source .venv/bin/activate
uv pip install -e .
```

3. **Install vllm**
```
cd deps/vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
cd ../..
```

## Usage

### Speculative Decoding Experiments
Configure model setups in `configs/sd_setups.yaml`, then run:
```bash
python scripts/run_sd.py \
  --config configs/sd_setups.yaml \
  --dataset chat \
  --num_prompts 500 \
  --setup_type few_setups \
  --output_dir results/sd_experiments \
  --input_tokens "4000"
```

To run with a specific context length in some range, use the `--input-tokens "min:max:step"`:


### Experiments with different VUS

Configure target and draft model setups in `configs/load_test.yaml`, then run:
```bash
python scripts/run_load_test.py --config configs/load_test.yaml
```

To run with a specific context length, use the `--input-tokens "min:max:step"`:
```bash
python scripts/run_load_test.py --config configs/load_test.yaml --input-tokens "2000:4100:1000"
```

### Data Analysis

Results are stored in the `results` directory. To analyze metrics using the database:

1. Import Speculative Decoding metrics:
```bash
python database/run.py \
  --etl_class sd_metrics \
  --data_dir results/sd_experiments/ \
  --db_name storage/database.db
```

2. Import load test metrics:
```bash
python database/run.py \
  --etl_class load_test_metrics \
  --data_dir scripts/load_test/artifacts/ \
  --db_name storage/database.db
```

3. To view the analysis results, go to `notebook.ipynb`.
