from pathlib import Path
from typing import Any, Dict

from lcspec.database.db import (
    get_dataset_id,
    get_model_id,
    get_sd_method_id,
    get_sd_setup_id,
    insert_dataset,
    insert_model,
    insert_sd_method,
    insert_sd_performance,
    insert_sd_setup,
)
from lcspec.database.etl.base import ETLBase


class SDMetrics(ETLBase):
    def __init__(self, db_name: str) -> None:
        super().__init__(db_name)

    def _extract(self, file_path: Path | str) -> Any:
        with open(file_path, "r") as f:
            data = f.read()
        return data

    def _transform(self, data: Any) -> Dict[Any, Any]:
        import json

        data = json.loads(data)

        target_model_name = data["main_model"]
        if target_model_name[-1] == "/":
            target_model_name = target_model_name[:-1]
        sd_model_name, sd_method_type = "", ""
        if data["speculative_model"]:
            sd_model_name = data["speculative_model"]
            if sd_model_name[-1] == "/":
                sd_model_name = sd_model_name[:-1]
            sd_method_type = data["method"]

        date = data["timestamp"]

        mean_acceptance_length = (
            data["mean_acceptance_length"] if data["mean_acceptance_length"] else 0
        )
        acceptance_rates = (
            data["acceptance_rates"]
            if data["acceptance_rates"]
            else [0 for _ in range(5)]
        )
        if len(acceptance_rates) < 5:
            acceptance_rates += [0.0] * (5 - len(acceptance_rates))

        assert len(acceptance_rates) == 5, "Acceptance rates must have 5 values."

        input_tokens = data["input_tokens"] if data["input_tokens"] else -1

        transformed_data = {
            "target_model": target_model_name,
            "sd_model": sd_model_name,
            "sd_method": sd_method_type,
            "dataset_type": data["dataset_type"],
            "time_taken": data["time_taken"],
            "date": date,
            "mean_acceptance_length": mean_acceptance_length,
            "acceptance_rates": acceptance_rates,
            "input_tokens": input_tokens,
        }
        return transformed_data

    def _load(self, data: Dict[Any, Any]) -> None:
        target_model_id = get_model_id(self.db_name, data["target_model"])
        if target_model_id is None:
            target_model_id = insert_model(self.db_name, data["target_model"])

        sd_model_id = get_model_id(self.db_name, data["sd_model"])
        if sd_model_id is None:
            sd_model_id = insert_model(self.db_name, data["sd_model"])

        sd_method_id = get_sd_method_id(self.db_name, data["sd_method"])
        if sd_method_id is None:
            sd_method_id = insert_sd_method(self.db_name, data["sd_method"])

        dataset_id = get_dataset_id(self.db_name, data["dataset_type"])
        if dataset_id is None:
            dataset_id = insert_dataset(self.db_name, data["dataset_type"])

        assert target_model_id is not None
        assert sd_model_id is not None
        assert sd_method_id is not None
        assert dataset_id is not None

        sd_setup_id = get_sd_setup_id(
            self.db_name,
            target_model_id,
            sd_model_id,
            sd_method_id,
            dataset_id,
        )
        if sd_setup_id is None:
            sd_setup_id = insert_sd_setup(
                self.db_name,
                target_model_id,
                sd_model_id,
                sd_method_id,
                dataset_id,
            )
        assert sd_setup_id is not None

        insert_sd_performance(
            self.db_name,
            sd_setup_id,
            data["mean_acceptance_length"],
            data["date"],
            data["time_taken"],
            data["acceptance_rates"],
            data["input_tokens"],
        )
