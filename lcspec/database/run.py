import argparse
import os
from pathlib import Path
from typing import Tuple, Type

from lcspec.database.db import create_database
from lcspec.database.etl.base import ETLBase
from lcspec.database.etl.load_test_metrics import LoadTestETL
from lcspec.database.etl.sd_metrics import SDMetrics
from lcspec.scripts.utils import setup_logger

logger = setup_logger(log_name="etl_process")


def get_etl_class_and_file_pattern(etl_name: str) -> Tuple[Type[ETLBase], str]:
    etl_classes = {
        "sd_metrics": SDMetrics,
        "load_test_metrics": LoadTestETL,
    }

    etl_file_patterns = {
        "sd_metrics": "sd_results_*.json",
        "load_test_metrics": "*",
    }

    if etl_name not in etl_classes:
        raise ValueError(
            f"Unknown ETL class: {etl_name}. Available classes: {list(etl_classes.keys())}"
        )

    return etl_classes[etl_name], etl_file_patterns[etl_name]


def process_files(
    etl_class: Type[ETLBase], data_dir: Path, db_name: str, file_pattern: str
) -> None:
    etl = etl_class(db_name)
    for file_path in data_dir.glob(file_pattern):
        if file_path.is_file():
            try:
                etl.run(file_path)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        elif file_path.is_dir() and etl_class.__name__ == "LoadTestETL":
            try:
                etl.run(file_path)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Run ETL process for database")
    parser.add_argument(
        "--etl_class",
        type=str,
        required=True,
        choices=["sd_metrics", "load_test_metrics"],
        help="ETL class to use (e.g., sd_metrics, load_test_metrics)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing data files to process",
    )
    parser.add_argument(
        "--db_name", type=str, required=True, help="SQLite database name"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    if not os.path.exists(args.db_name):
        logger.info(f"Creating new database: {args.db_name}")
        create_database(args.db_name)

    try:
        etl_class, file_pattern = get_etl_class_and_file_pattern(args.etl_class)
        process_files(etl_class, data_dir, args.db_name, file_pattern)
    except ValueError as e:
        logger.error(e)


if __name__ == "__main__":
    main()
