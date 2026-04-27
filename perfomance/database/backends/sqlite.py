import logging
import sqlite3
from pathlib import Path

from perfomance.scripts.utils import load_config

from .base import BaseDBBackend

logger = logging.getLogger(__name__)

_TABLES_YAML = Path(__file__).parent.parent / "tables.yaml"


class SQLiteBackend(BaseDBBackend):
    _ph = "?"

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")

    def _cursor(self):
        return self._conn.cursor()

    def _commit(self) -> None:
        self._conn.commit()

    def create_tables(self) -> None:
        tables = load_config(_TABLES_YAML).get("database_tables", {})
        try:
            for table_name, values in tables.items():
                columns = values["columns"]
                dependent_columns = values.get("dependent_columns", {})
                unique_constraints = values.get("unique_constraints", [])

                column_defs = [
                    f"{col_name} {col_type}"
                    for col_name, col_type in columns.items()
                ]
                fk_defs = [
                    f"FOREIGN KEY ({col_name}) REFERENCES {ref_table}"
                    for col_name, ref_table in dependent_columns.items()
                ]
                unique_defs = [
                    f"UNIQUE ({', '.join(cols)})"
                    for cols in unique_constraints
                ]
                self._conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table_name}"
                    f" ({', '.join(column_defs + fk_defs + unique_defs)})"
                )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error creating SQLite tables: {e}")
