import sqlite3
from pathlib import Path
from typing import List, Optional

from lcspec.scripts.utils import load_config


def create_database(db_name: str) -> None:
    """Create SQLite database and initialize tables from YAML definitions"""
    conn = sqlite3.connect(db_name)
    conn.execute("PRAGMA foreign_keys = ON")

    current_dir = Path(__file__).parent
    tables = load_config(current_dir / "tables.yaml").get("database_tables", {})
    try:
        for table_name, values in tables.items():
            columns = values["columns"]
            dependent_columns = values.get("dependent_columns", {})

            column_defs = [
                f"{col_name} {col_type}" for col_name, col_type in columns.items()
            ]
            dependent_column_defs = [
                f" FOREIGN KEY ({col_name}) REFERENCES {col_base_table}"
                for col_name, col_base_table in dependent_columns.items()
            ]

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {", ".join(column_defs + dependent_column_defs)}
            )
            """

            conn.execute(create_table_sql)

        conn.commit()
    except sqlite3.Error as e:
        print(f"Error creating tables: {e}")
    finally:
        conn.close()


def insert_model(db_name: str, model_name: str) -> Optional[int]:
    """Insert a row into models table"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "INSERT INTO models (model_name) VALUES (?)", (model_name,)
        )
        model_id = cursor.lastrowid
        conn.commit()
        return model_id
    finally:
        conn.close()


def insert_sd_method(db_name: str, sd_method_type: str) -> Optional[int]:
    """Insert a row into sd_methods table"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "INSERT INTO sd_methods (sd_method_type) VALUES (?)",
            (sd_method_type,),
        )
        sd_method_id = cursor.lastrowid
        conn.commit()
        return sd_method_id
    finally:
        conn.close()


def insert_dataset(db_name: str, dataset_type: str) -> Optional[int]:
    """Insert a row into datasets table"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "INSERT INTO datasets (dataset_type) VALUES (?)",
            (dataset_type,),
        )
        dataset_id = cursor.lastrowid
        conn.commit()
        return dataset_id
    finally:
        conn.close()


def insert_sd_setup(
    db_name: str,
    target_model_id: int,
    sd_model_id: int,
    sd_method_id: int,
    dataset_id: int,
) -> Optional[int]:
    """Insert a row into sd_setups table"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            """INSERT INTO sd_setups
            (target_model_id, sd_model_id, sd_method_id, dataset_id)
            VALUES (?, ?, ?, ?)""",
            (
                target_model_id,
                sd_model_id,
                sd_method_id,
                dataset_id,
            ),
        )
        sd_setup_id = cursor.lastrowid
        conn.commit()
        return sd_setup_id
    finally:
        conn.close()


def insert_load_test_performance(
    db_name: str,
    sd_setup_id: int,
    load_value: int,
    end2end: float,
    input_tokens: int,
    num_spec_tokens: int,
    date: str,
) -> None:
    """Insert a row into ld_performances table"""
    conn = sqlite3.connect(db_name)
    try:
        conn.execute(
            "INSERT INTO ld_performances (sd_setup_id, load, end_to_end_latency, input_tokens, num_spec_tokens, date) VALUES (?, ?, ?, ?, ?, ?)",
            (sd_setup_id, load_value, end2end, input_tokens, num_spec_tokens, date),
        )
        conn.commit()
    finally:
        conn.close()


def insert_sd_performance(
    db_name: str,
    sd_setup_id: int,
    mean_acceptance_length: float,
    date: str,
    time_taken: float,
    acceptance_rates: List[float],
    input_tokens: int = -1,
) -> None:
    """Insert a row into sd_performances table"""
    conn = sqlite3.connect(db_name)
    try:
        if len(acceptance_rates) < 5:
            acceptance_rates += [0.0] * (5 - len(acceptance_rates))
        ar_1, ar_2, ar_3, ar_4, ar_5 = acceptance_rates
        conn.execute(
            "INSERT INTO sd_performances (date, sd_setup_id, mean_acceptance_length, time_taken, rate_at_1position, rate_at_2position, rate_at_3position, rate_at_4position, rate_at_5position, input_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                date,
                sd_setup_id,
                mean_acceptance_length,
                time_taken,
                ar_1,
                ar_2,
                ar_3,
                ar_4,
                ar_5,
                input_tokens,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_model_id(db_name: str, model_name: str) -> Optional[int]:
    """Search for model by name and return its ID if exists"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", (model_name,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()


def get_sd_method_id(db_name: str, sd_method_type: str) -> Optional[int]:
    """Search for sd_method by type and return its ID if exists"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "SELECT sd_method_id FROM sd_methods WHERE sd_method_type = ?",
            (sd_method_type,),
        )
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()


def get_dataset_id(db_name: str, dataset_type: str) -> int | None:
    """Search for dataset by type and return its ID if exists"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "SELECT dataset_id FROM datasets WHERE dataset_type = ?",
            (dataset_type,),
        )
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()


def get_sd_setup_id(
    db_name: str,
    target_model_id: int,
    sd_model_id: int,
    sd_method_id: int,
    dataset_id: int,
) -> int | None:
    """Search for SD setup by IDs and return its ID if exists"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            """SELECT sd_setup_id FROM sd_setups
            WHERE target_model_id = ? AND sd_model_id = ?
            AND sd_method_id = ? AND dataset_id = ?
            """,
            (
                target_model_id,
                sd_model_id,
                sd_method_id,
                dataset_id,
            ),
        )
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()
