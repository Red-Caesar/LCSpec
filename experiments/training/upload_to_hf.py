import argparse
from pathlib import Path

from base import (
    BaselineConfig,
    DynamicYarnConfig,
    FinetunedBaseConfig,
    FinetunedYarnConfig,
    PartialRopeConfig,
    TrainingMethod,
)
from huggingface_hub import HfApi

OUTPUT_BASE = Path("./output")

METHODS: dict[str, TrainingMethod] = {
    "baseline": BaselineConfig(),
    "yarn": DynamicYarnConfig(),
    "partial": PartialRopeConfig(),
    "finetuned_base": FinetunedBaseConfig(),
    "finetuned_yarn": FinetunedYarnConfig(),
}

_SKIP_PATTERNS = ["optimizer_state_dict.pt", "scheduler_state_dict.pt"]


def latest_checkpoint(checkpoints_dir: Path) -> Path:
    candidates = [p for p in checkpoints_dir.iterdir() if p.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    return max(candidates, key=lambda p: int(p.name))


def upload(
    method: TrainingMethod,
    repo_id: str,
    checkpoint: int | None,
    save_state_dicts: bool,
) -> None:
    checkpoints_dir = OUTPUT_BASE / method.output_subdir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    if checkpoint is not None:
        checkpoint_dir = checkpoints_dir / str(checkpoint)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint {checkpoint} not found: {checkpoint_dir}"
            )
    else:
        checkpoint_dir = latest_checkpoint(checkpoints_dir)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Uploading {checkpoint_dir} → {repo_id}")
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=_SKIP_PATTERNS if save_state_dicts else [],
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload speculator checkpoint to HuggingFace."
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        required=True,
        help="Training method whose checkpoint to upload.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo ID, e.g. username/model-name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint number to upload. Defaults to the latest.",
    )
    parser.add_argument(
        "--save_state_dicts",
        type=bool,
        action="store_true",
    )
    args = parser.parse_args()
    upload(METHODS[args.method], args.repo, args.checkpoint, args.save_state_dicts)
