# Training

Trains an [EAGLE3](https://github.com/SafeAILab/EAGLE) speculative decoding draft model on top of `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` using the [speculators](../../deps/speculators) library.

Four RoPE strategies are available, each controlling how the draft model handles long-context position encodings:

| Method | Description |
|---|---|
| `baseline` | Standard Eagle3, no RoPE modification |
| `yarn` | Dynamic YaRN RoPE scaling (factor 16×) |
| `partial` | Qwen3-style partial RoPE (25% of dimensions rotated) |

## Run training

```bash
cd experiments/training
python train.py --method <method>
```

Checkpoints are saved to `output/<method>/checkpoints/`.

## Fine-tune a pre-trained Eagle3 checkpoint

Instead of training from scratch, you can start from an existing Eagle3 checkpoint (e.g. `RedCaesar/EAGLE3-LLaMA3.1-Instruct-8B`) and fine-tune it on long-context data. This is faster and preserves general speculator quality while adapting the model to longer sequences.

Two modes are available:

| Method | Description |
|---|---|
| `base` | Fine-tune on long-context data with standard RoPE |
| `yarn` | Apply YaRN RoPE scaling before fine-tuning |

```bash
cd experiments/training
python finetune_pretrained.py --method <method>
```

Checkpoints are saved to `output/finetune_<method>/checkpoints/`.

## Upload a checkpoint to HuggingFace

```bash
python upload_to_hf.py --method <method> --repo <hf-username>/<repo-name>
```

By default uploads the latest checkpoint. To upload a specific one:

```bash
python upload_to_hf.py --method <method> --repo <hf-username>/<repo-name> --checkpoint <number>
```
