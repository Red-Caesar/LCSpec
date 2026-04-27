# Training

Trains an [EAGLE3](https://github.com/SafeAILab/EAGLE) speculative decoding draft model on top of `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` using the [speculators](../../deps/speculators) library.

Four RoPE strategies are available, each controlling how the draft model handles long-context position encodings:

| Method | Description |
|---|---|
| `baseline` | Standard Eagle3, no RoPE modification |
| `yarn` | Dynamic YaRN RoPE scaling (factor 4×) |
| `partial` | Qwen3-style partial RoPE (25% of dimensions rotated) |

## Run training

```bash
cd experiments/training
python train.py --method <method>
```

Checkpoints are saved to `output/<method>/checkpoints/`.

## Upload a checkpoint to HuggingFace

```bash
python upload_to_hf.py --method <method> --repo <hf-username>/<repo-name>
```

By default uploads the latest checkpoint. To upload a specific one:

```bash
python upload_to_hf.py --method <method> --repo <hf-username>/<repo-name> --checkpoint <number>
```
