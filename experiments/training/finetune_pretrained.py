import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

sys.path.insert(
    0, str(Path(__file__).absolute().parents[2] / "deps/speculators/scripts")
)
from gen_and_train import DataGenArgs, TrainArgs, VocabMappingArgs, run_e2e  # noqa: E402

VERIFIER = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
DEFAULT_PRETRAINED_PATH = "/workspace/workflow/open_model/RedCaesar/EAGLE3-LLaMA3.1-Instruct-8B"
OUTPUT_BASE = Path("./output")

TARGET_VOCAB_SIZE = 128256
DRAFT_VOCAB_SIZE = 32000

SHAREGPT_SAMPLES_PER_WAVE = 3000
LONGALPACA_SAMPLES_PER_WAVE = 100
GPU_MEMORY_UTILIZATION = 0.9
NUM_PREPROCESSING_WORKERS = 16
SHAREGPT_BATCH_SIZE = 32
LONGALPACA_BATCH_SIZE = 4
TRAIN_SEQ_LEN = 2048

YARN_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 2048,
}


def convert_weights(pretrained_path: Path, checkpoint_path: Path) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    target = checkpoint_path / "model.safetensors"

    if target.exists():
        print(f"Checkpoint already exists at {target}, skipping conversion.")
        return

    src = pretrained_path / "model.safetensors"
    if not src.exists():
        raise FileNotFoundError(f"Pre-trained weights not found at {src}")

    print(f"Converting weights from {src} → {target}")
    state_dict = load_file(str(src))

    converted = {}
    for key, tensor in state_dict.items():
        if key in ("d2t", "t2d"):
            continue
        if key.startswith("midlayer."):
            new_key = "layers.0." + key[len("midlayer."):]
            converted[new_key] = tensor
        else:
            converted[key] = tensor

    print("Converted keys:")
    for k in sorted(converted):
        print(f"  {k}")

    save_file(converted, str(target))
    print(f"Saved converted weights to {target}")


def build_data_gen_args(seed: int) -> list[DataGenArgs]:
    return [
        DataGenArgs(
            train_data_path="sharegpt",
            seq_length=TRAIN_SEQ_LEN,
            turn_dropout=True,
            batch_size=SHAREGPT_BATCH_SIZE,
            max_samples=SHAREGPT_SAMPLES_PER_WAVE,
            seed=seed,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            num_preprocessing_workers=NUM_PREPROCESSING_WORKERS,
        ),
        DataGenArgs(
            train_data_path="longalpaca",
            dataset_name="longalpaca",
            seq_length=TRAIN_SEQ_LEN,
            turn_dropout=False,
            batch_size=LONGALPACA_BATCH_SIZE,
            max_samples=LONGALPACA_SAMPLES_PER_WAVE,
            seed=seed,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            num_preprocessing_workers=NUM_PREPROCESSING_WORKERS,
        ),
    ]


def build_train_args(mode: str, epochs: int) -> TrainArgs:
    common = dict(
        logger="tensorboard",
        lr=2e-5,
        total_seq_len=TRAIN_SEQ_LEN,
        epochs=epochs,
        num_layers=1,
        ttt_steps=3,
        use_off_policy_tokens=True,
        scheduler_type="none",
        speculator_type="eagle3",
    )
    if mode == "yarn":
        return TrainArgs(
            run_name="finetune_yarn_llama31_8b",
            rope_scaling=json.dumps(YARN_ROPE_SCALING),
            **common,
        )
    return TrainArgs(
        run_name="finetune_base_llama31_8b",
        **common,
    )


def run_waves(args: argparse.Namespace) -> None:
    output_path = OUTPUT_BASE / f"finetune_{args.mode}"
    checkpoint_epoch0 = output_path / "checkpoints" / "0"

    convert_weights(Path(args.pretrained_path), checkpoint_epoch0)

    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
    )

    gen_dir = output_path / "gen"

    for wave in range(args.num_waves):
        data_gen_args = build_data_gen_args(seed=wave)
        train_args = build_train_args(args.mode, epochs=(wave + 1))

        run_e2e(
            verifier_name_or_path=args.verifier,
            output_path=str(output_path),
            data_gen_args=data_gen_args,
            vocab_mapping_args=vocab_mapping_args if wave == 0 else None,
            train_args=train_args,
        )

        if gen_dir.exists() and wave < args.num_waves - 1:
            shutil.rmtree(gen_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained Eagle3 model on long-context data."
    )
    parser.add_argument(
        "--mode",
        choices=["base", "yarn"],
        required=True,
        help=(
            "base: fine-tune on long context with standard RoPE. "
            "yarn: apply YaRN RoPE scaling before fine-tuning."
        ),
    )
    parser.add_argument(
        "--pretrained-path",
        default=DEFAULT_PRETRAINED_PATH,
        help="Path to the pre-trained Eagle3 model directory.",
    )
    parser.add_argument(
        "--verifier",
        default=VERIFIER,
        help="Verifier model name or path (HuggingFace or local).",
    )
    parser.add_argument(
        "--num-waves",
        type=int,
        default=2,
        help="Number of training waves (each wave = 1 epoch on fresh data).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_waves(parse_args())
