import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).absolute().parents[2] / "deps/speculators/scripts")
)
from gen_and_train import DataGenArgs, TrainArgs

TRAIN_SEQ_LEN = 8192

SHAREGPT_SEQ_LEN = TRAIN_SEQ_LEN
ULTRACHAT_SEQ_LEN = TRAIN_SEQ_LEN
LONGALPACA_SEQ_LEN = TRAIN_SEQ_LEN

SHAREGPT_BATCH_SIZE = 32
ULTRACHAT_BATCH_SIZE = 32
LONGALPACA_BATCH_SIZE = 4

GPU_MEMORY_UTILIZATION = 0.9
NUM_PREPROCESSING_WORKERS = 16


class TrainingMethod:
    output_subdir: str
    run_name: str
    _train_seq_len: int = TRAIN_SEQ_LEN
    _speculator_type: str | None = None
    _rope_method: str | None = None
    _rope_scaling: str | None = None
    _partial_rotary_factor: float | None = None

    def build_data_gen_args(
        self, seed: int, sharegpt_samples: int, ultrachat_samples: int, longalpaca_samples: int
    ) -> list[DataGenArgs]:
        return [
            DataGenArgs(
                train_data_path="sharegpt",
                seq_length=SHAREGPT_SEQ_LEN,
                turn_dropout=True,
                batch_size=SHAREGPT_BATCH_SIZE,
                max_samples=sharegpt_samples,
                seed=seed,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                num_preprocessing_workers=NUM_PREPROCESSING_WORKERS,
            ),
            DataGenArgs(
                train_data_path="ultrachat",
                seq_length=ULTRACHAT_SEQ_LEN,
                turn_dropout=True,
                batch_size=ULTRACHAT_BATCH_SIZE,
                max_samples=ultrachat_samples,
                seed=seed,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                num_preprocessing_workers=NUM_PREPROCESSING_WORKERS,
            ),
            DataGenArgs(
                train_data_path="longalpaca",
                dataset_name="longalpaca",
                seq_length=LONGALPACA_SEQ_LEN,
                turn_dropout=False,
                batch_size=LONGALPACA_BATCH_SIZE,
                max_samples=longalpaca_samples,
                seed=seed,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                num_preprocessing_workers=NUM_PREPROCESSING_WORKERS,
            ),
        ]

    def build_train_args(self, epochs: int) -> TrainArgs:
        kwargs: dict = dict(
            run_name=self.run_name,
            logger="tensorboard",
            lr=2e-5,
            total_seq_len=self._train_seq_len,
            epochs=epochs,
            num_layers=1,
            ttt_steps=3,
            use_off_policy_tokens=True,
            scheduler_warmup_steps=200,
            scheduler_type="none",
        )
        for attr, key in (
            ("_speculator_type", "speculator_type"),
            ("_rope_scaling", "rope_scaling"),
            ("_partial_rotary_factor", "partial_rotary_factor"),
        ):
            val = getattr(self, attr)
            if val is not None:
                kwargs[key] = val
        return TrainArgs(**kwargs)


class BaselineConfig(TrainingMethod):
    """Standard Eagle3 baseline without RoPE modifications."""

    output_subdir = "baseline"
    run_name = "baseline_llama31_8b_fp8"


class DynamicYarnConfig(TrainingMethod):
    """Eagle3-LC with dynamic YaRN RoPE scaling."""

    output_subdir = "dynamic_yarn"
    run_name = "dynamic_yarn_llama31_8b_fp8"
    _speculator_type = "eagle3"
    _rope_scaling = json.dumps(
        {"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 2048}
    )


class Llama3RopeConfig(TrainingMethod):
    """Eagle3-LC with Llama-3.1 high/low-frequency RoPE interpolation."""

    output_subdir = "llama3rope"
    run_name = "llama3rope_llama31_8b_fp8"
    _speculator_type = "eagle3"
    _rope_scaling = json.dumps(
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 2048,
        }
    )


class PartialRopeConfig(TrainingMethod):
    """Eagle3-LC with Qwen3-style partial RoPE."""

    output_subdir = "partial_rope"
    run_name = "partial_rope_llama31_8b_fp8"
    _speculator_type = "eagle3"
    _partial_rotary_factor = 0.25
