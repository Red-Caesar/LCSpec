import argparse
import shutil
import sys
from pathlib import Path

from base import FinetunedBaseConfig, FinetunedYarnConfig, TrainingMethod
from safetensors.torch import load_file, save_file

sys.path.insert(
    0, str(Path(__file__).absolute().parents[2] / "deps/speculators/scripts")
)
from gen_and_train import VocabMappingArgs, run_e2e  # noqa: E402

VERIFIER = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
DEFAULT_PRETRAINED_PATH = (
    "/workspace/workflow/open_model/RedCaesar/EAGLE3-LLaMA3.1-Instruct-8B"
)
OUTPUT_BASE = Path("./output")

TARGET_VOCAB_SIZE = 128256
DRAFT_VOCAB_SIZE = 32000

SHAREGPT_SAMPLES_PER_WAVE = 3000
LONGALPACA_SAMPLES_PER_WAVE = 500

METHODS: dict[str, TrainingMethod] = {
    "base": FinetunedBaseConfig(),
    "yarn": FinetunedYarnConfig(),
}

dataset_samples: dict[str, int] = {
    "sharegpt": SHAREGPT_SAMPLES_PER_WAVE,
    "longalpaca": LONGALPACA_SAMPLES_PER_WAVE,
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
            new_key = "layers.0." + key[len("midlayer.") :]
            converted[new_key] = tensor
        else:
            converted[key] = tensor

    print("Converted keys:")
    for k in sorted(converted):
        print(f"  {k}")

    save_file(converted, str(target))
    print(f"Saved converted weights to {target}")


def run_waves(args: argparse.Namespace) -> None:
    method = METHODS[args.method]
    output_path = OUTPUT_BASE / method.output_subdir
    checkpoint_epoch0 = output_path / "checkpoints" / "0"

    convert_weights(Path(args.pretrained_path), checkpoint_epoch0)

    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
    )

    gen_dir = output_path / "gen"

    for wave in range(args.num_waves):
        data_gen_args = method.build_data_gen_args(
            seed=wave,
            dataset_samples=dataset_samples,
        )
        train_args = method.build_train_args(epochs=(wave + 1))
        run_e2e(
            verifier_name_or_path=VERIFIER,
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
        "--method",
        choices=list(METHODS),
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
        "--num-waves",
        type=int,
        default=2,
        help="Number of training waves (each wave = 1 epoch on fresh data).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_waves(parse_args())
