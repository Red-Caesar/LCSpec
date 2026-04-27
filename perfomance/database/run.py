import argparse
from pathlib import Path
from typing import Tuple, Type

from perfomance.database.backends.postgres import PostgresBackend
from perfomance.database.backends.sqlite import SQLiteBackend
from perfomance.database.db import DatabaseClient
from perfomance.database.etl.base import ETLBase
from perfomance.database.etl.load_test_metrics import LoadTestETL
from perfomance.database.etl.sd_metrics import SDMetrics
from perfomance.database.supabase import connect_supabase
from perfomance.scripts.utils import setup_logger

logger = setup_logger(log_name="etl_process")

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"


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
    etl_class: Type[ETLBase], data_dir: Path, db: DatabaseClient, file_pattern: str
) -> None:
    etl = etl_class(db)
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
        "--db_name",
        type=str,
        required=True,
        help="SQLite database filename (always stored in project storage/ directory)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    sqlite_path = str(STORAGE_DIR / args.db_name)

    pg_conn = connect_supabase()
    local = SQLiteBackend(sqlite_path)
    remote = PostgresBackend(pg_conn) if pg_conn is not None else None
    db = DatabaseClient(local, remote)

    logger.info(f"Using local database: {sqlite_path}")
    db.create_tables()

    try:
        etl_class, file_pattern = get_etl_class_and_file_pattern(args.etl_class)
        process_files(etl_class, data_dir, db, file_pattern)
    except ValueError as e:
        logger.error(e)
    finally:
        if pg_conn is not None:
            pg_conn.close()


if __name__ == "__main__":
    main()
