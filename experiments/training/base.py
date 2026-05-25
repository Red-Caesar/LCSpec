import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).absolute().parents[2] / "deps/speculators/scripts")
)
from typing import Dict, List

from gen_and_train import DataGenArgs, TrainArgs

TRAIN_SEQ_LEN = 8192

SHAREGPT_SEQ_LEN = TRAIN_SEQ_LEN
ULTRACHAT_SEQ_LEN = TRAIN_SEQ_LEN
LONGALPACA_SEQ_LEN = TRAIN_SEQ_LEN
LONGALIGN_SEQ_LEN = TRAIN_SEQ_LEN

SHAREGPT_BATCH_SIZE = 32
ULTRACHAT_BATCH_SIZE = 32
LONGALPACA_BATCH_SIZE = 4
LONGALIGN_BATCH_SIZE = 4

GPU_MEMORY_UTILIZATION = 0.9
NUM_PREPROCESSING_WORKERS = 16

DATASETS = {
    "sharegpt": {
        "train_data_path": "sharegpt",
        "seq_length": SHAREGPT_SEQ_LEN,
        "turn_dropout": True,
        "batch_size": SHAREGPT_BATCH_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
    },
    "ultrachat": {
        "train_data_path": "ultrachat",
        "seq_length": ULTRACHAT_SEQ_LEN,
        "turn_dropout": True,
        "batch_size": ULTRACHAT_BATCH_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
    },
    "longalpaca": {
        "train_data_path": "longalpaca",
        "dataset_name": "longalpaca",
        "seq_length": LONGALPACA_SEQ_LEN,
        "turn_dropout": False,
        "batch_size": LONGALPACA_BATCH_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
    },
    "longalign": {
        "train_data_path": "longalign",
        "dataset_name": "longalign",
        "seq_length": LONGALIGN_SEQ_LEN,
        "turn_dropout": False,
        "batch_size": LONGALIGN_BATCH_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
    },
}


class TrainingMethod:
    output_subdir: str
    run_name: str
    _datasets_to_use: List[str]
    _train_seq_len: int = TRAIN_SEQ_LEN
    _rope_scaling: str | None = None
    _partial_rotary_factor: float | None = None
    _scheduler_warmup_steps: int | None = None

    def build_data_gen_args(
        self, seed: int, dataset_samples: Dict[str, int]
    ) -> list[DataGenArgs]:
        choosen_datasets = []
        for dataset_name in self._datasets_to_use:
            base_data_args = DATASETS[dataset_name].copy()
            base_data_args["max_samples"] = dataset_samples[dataset_name]
            base_data_args["seed"] = seed
            choosen_datasets.append(DataGenArgs(**base_data_args))
        return choosen_datasets

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
            scheduler_type="none",
            speculator_type="eagle3",
        )
        for attr, key in (
            ("_scheduler_warmup_steps", "scheduler_warmup_steps"),
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
    _datasets_to_use = ["sharegpt", "ultrachat", "longalpaca", "longalign"]


class DynamicYarnConfig(TrainingMethod):
    """Eagle3 with dynamic YaRN RoPE scaling."""

    output_subdir = "dynamic_yarn"
    run_name = "dynamic_yarn_llama31_8b_fp8"
    _scheduler_warmup_steps = 200
    _rope_scaling = json.dumps(
        {"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 2048}
    )
    _datasets_to_use = ["sharegpt", "ultrachat", "longalpaca", "longalign"]


class Llama3RopeConfig(TrainingMethod):
    """Eagle3 with Llama-3.1 high/low-frequency RoPE interpolation."""

    output_subdir = "llama3_rope"
    run_name = "llama3_rope_llama31_8b_fp8"
    _scheduler_warmup_steps = 200
    _rope_scaling = json.dumps(
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 2048,
        }
    )
    _datasets_to_use = ["sharegpt", "ultrachat", "longalpaca", "longalign"]


class PartialRopeConfig(TrainingMethod):
    """Eagle3 with Qwen3-style partial RoPE."""

    output_subdir = "partial_rope"
    run_name = "partial_rope_llama31_8b_fp8"
    _scheduler_warmup_steps = 200
    _partial_rotary_factor = 0.25
    _datasets_to_use = ["sharegpt", "ultrachat", "longalpaca", "longalign"]


class FinetunedYarnConfig(TrainingMethod):
    output_subdir = "finetune_yarn"
    run_name = "finetune_yarn_llama31_8b"
    _rope_scaling = json.dumps(
        {"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 2048}
    )
    _datasets_to_use = ["sharegpt", "longalpaca"]


class FinetunedBaseConfig(TrainingMethod):
    output_subdir = "finetune_base"
    run_name = "finetune_base_llama31_8b"
    _datasets_to_use = ["sharegpt", "longalpaca"]
