from pathlib import Path
from typing import Any, Dict

from perfomance.database.db import DatabaseClient
from perfomance.database.etl.base import ETLBase


class SDMetrics(ETLBase):
    def __init__(self, db: DatabaseClient) -> None:
        super().__init__(db)

    def _extract(self, file_path: Path | str) -> Any:
        with open(file_path, "r") as f:
            data = f.read()
        return data

    def _transform(self, data: Any) -> Dict[Any, Any]:
        import json

        data = json.loads(data)

        target_model_name = data["main_model"].rstrip("/")
        sd_model_name, sd_method_type = "", ""
        if data["speculative_model"]:
            sd_model_name = data["speculative_model"].rstrip("/")
            sd_method_type = "sd_" + data["method"]

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
        local_id, remote_id = self.db.resolve_sd_setup(data)
        self.db.insert_sd_performance(
            local_id,
            data["mean_acceptance_length"],
            data["date"],
            data["time_taken"],
            data["acceptance_rates"],
            data["input_tokens"],
            remote_id=remote_id,
        )
