import argparse
import shutil
import sys
from pathlib import Path

from base import (
    BaselineConfig,
    DynamicYarnConfig,
    Llama3RopeConfig,
    PartialRopeConfig,
    TrainingMethod,
)

sys.path.insert(
    0, str(Path(__file__).absolute().parents[2] / "deps/speculators/scripts")
)

from gen_and_train import VocabMappingArgs, run_e2e  # noqa: E402

VERIFIER = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
OUTPUT_BASE = Path("./output")

TARGET_VOCAB_SIZE = 128256
DRAFT_VOCAB_SIZE = 32000

NUM_WAVES = 10
EPOCHS_PER_WAVE = 1
# median 1518 tokens
SHAREGPT_SAMPLES_PER_WAVE = 3000
# median 1108 tokens
ULTRACHAT_SAMPLES_PER_WAVE = 4000
# median 7888 tokens
LONGALPACA_SAMPLES_PER_WAVE = 100
# median 12690 tokens
LONGALIGN_SAMPLES_PER_WAVE = 10

dataset_samples: dict[str, int] = {
    "sharegpt": SHAREGPT_SAMPLES_PER_WAVE,
    "ultrachat": ULTRACHAT_SAMPLES_PER_WAVE,
    "longalpaca": LONGALPACA_SAMPLES_PER_WAVE,
    "longalign": LONGALIGN_SAMPLES_PER_WAVE,
}

METHODS: dict[str, TrainingMethod] = {
    "baseline": BaselineConfig(),
    "yarn": DynamicYarnConfig(),
    "llama3rope": Llama3RopeConfig(),
    "partial": PartialRopeConfig(),
}


def run_waves(method: TrainingMethod) -> None:
    output_path = OUTPUT_BASE / method.output_subdir
    gen_dir = output_path / "gen"
    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
    )

    for wave in range(NUM_WAVES):
        data_gen_args = method.build_data_gen_args(
            seed=wave,
            dataset_samples=dataset_samples,
        )
        train_args = method.build_train_args(epochs=(wave + 1) * EPOCHS_PER_WAVE)
        run_e2e(
            verifier_name_or_path=VERIFIER,
            output_path=str(output_path),
            data_gen_args=data_gen_args,
            vocab_mapping_args=vocab_mapping_args if wave == 0 else None,
            train_args=train_args,
        )
        if gen_dir.exists() and wave < NUM_WAVES - 1:
            shutil.rmtree(gen_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        required=True,
        help="Training method to run.",
    )
    args = parser.parse_args()
    run_waves(METHODS[args.method])
