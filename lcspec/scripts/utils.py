import logging
from pathlib import Path
from typing import Union

import yaml

LOG_PATH = Path(".logs")


def setup_logger(
    output_dir: Union[str, Path] = LOG_PATH, log_name: str = "test"
) -> logging.Logger:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir) / f"{log_name}.log"
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)