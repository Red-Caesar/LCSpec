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
