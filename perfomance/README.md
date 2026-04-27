# Perfomance

## Usage

### Speculative Decoding Experiments
Configure model setups in `configs/sd_setups.yaml`, then run:
```bash
python scripts/run_sd.py \
  --config configs/sd_setups.yaml \
  --dataset chat \
  --num-prompts 500 \
  --setup-type few_setups \
  --output-dir results/sd_experiments \
  --input-tokens "4000"
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
