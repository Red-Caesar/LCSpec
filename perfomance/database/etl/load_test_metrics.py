import json
from pathlib import Path
from typing import Any, Dict

from perfomance.database.db import DatabaseClient
from perfomance.database.etl.base import ETLBase
from perfomance.scripts.utils import normalize_model_name


class LoadTestETL(ETLBase):
    def __init__(self, db: DatabaseClient) -> None:
        super().__init__(db)

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

        target_model_name = normalize_model_name(input_params.get("server", {}).get("model", ""))
        run_id = input_params.get("run_id", "")
        num_spec_tokens = 0
        sd_model_name = ""
        sd_method = ""
        sd_type = ""
        if "sd" in run_id:
            sd_type, sd_model_name, sd_method, num_spec_tokens = run_id.split("#")
            sd_type += "_"

        dataset_type = input_params.get("workload", {}).get("dataset", "")
        raw_date = "_".join(data["folder_name"].split("_")[-2:])
        date_part, time_part = raw_date.split("_")
        date = f"{date_part} {time_part.replace('-', ':')}"
        end2end = metrics.get("summary", {}).get("end-to-end", {}).get("median", 0.0)
        transformed = {
            "target_model": target_model_name,
            "sd_model": sd_model_name,
            "sd_method": sd_type + sd_method,
            "dataset_type": dataset_type,
            "load": load_value,
            "end_to_end_latency": end2end,
            "input_tokens": input_tokens,
            "num_spec_tokens": int(num_spec_tokens),
            "date": date,
        }
        return transformed

    def _load(self, data: Dict[Any, Any]) -> None:
        local_id, remote_id = self.db.resolve_sd_setup(data)
        self.db.insert_load_test_performance(
            local_id,
            data["load"],
            data["end_to_end_latency"],
            data["input_tokens"],
            data["num_spec_tokens"],
            data["date"],
            remote_id=remote_id,
        )
