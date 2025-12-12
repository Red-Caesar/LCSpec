import json
from pathlib import Path
from typing import Any, Dict

from lcspec.database.db import (
    get_dataset_id,
    get_model_id,
    get_sd_method_id,
    get_sd_setup_id,
    insert_dataset,
    insert_load_test_performance,
    insert_model,
    insert_sd_method,
    insert_sd_setup,
)
from lcspec.database.etl.base import ETLBase


class LoadTestETL(ETLBase):
    def __init__(self, db_name: str) -> None:
        super().__init__(db_name)

    def _extract(self, folder_path: Path | str) -> Any:
        folder = Path(folder_path)
        with open(folder / "config.json", "r") as f:
            input_params = json.load(f)
        with open(folder / "summary.json", "r") as f:
            metrics = json.load(f)
        return {
            "input_params": input_params,
            "metrics": metrics,
            "folder_name": folder.name,
        }

    def _transform(self, data: Any) -> Dict[Any, Any]:
        input_params = data["input_params"]
        metrics = data["metrics"]

        load = input_params.get("workload", {}).get("load", "")
        load_value = int(load.split("=")[1])

        input_tokens = int(input_params.get("workload", {}).get("context_len", ""))

        target_model_name = input_params.get("server", {}).get("model", "")
        if target_model_name[-1] == "/":
            target_model_name = target_model_name[:-1]
        target_model_name = target_model_name.split("/")[-1]
        run_id = input_params.get("run_id", "")
        num_spec_tokens = 0
        sd_model_name = ""
        sd_method = ""
        if "sd#" in run_id:
            sd_model_name, sd_method, num_spec_tokens = run_id.split("#")[1:]

        dataset_type = input_params.get("workload", {}).get("dataset", "")
        date = "_".join(data["folder_name"].split("_")[-2:])
        end2end = metrics.get("summary", {}).get("end-to-end", {}).get("median", 0.0)
        transformed = {
            "target_model": target_model_name,
            "sd_model": sd_model_name,
            "sd_method": sd_method,
            "dataset_type": dataset_type,
            "load": load_value,
            "end_to_end_latency": end2end,
            "input_tokens": input_tokens,
            "num_spec_tokens": int(num_spec_tokens),
            "date": date,
        }
        return transformed

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
        insert_load_test_performance(
            self.db_name,
            sd_setup_id,
            data["load"],
            data["end_to_end_latency"],
            data["input_tokens"],
            data["num_spec_tokens"],
            data["date"],
        )
