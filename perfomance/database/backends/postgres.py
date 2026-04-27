import logging

from .base import BaseDBBackend

logger = logging.getLogger(__name__)

_PG_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS models (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS sd_methods (
    sd_method_id SERIAL PRIMARY KEY,
    sd_method_type TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id SERIAL PRIMARY KEY,
    dataset_type TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS sd_setups (
    sd_setup_id SERIAL PRIMARY KEY,
    target_model_id INTEGER REFERENCES models(model_id),
    sd_model_id INTEGER REFERENCES models(model_id),
    sd_method_id INTEGER REFERENCES sd_methods(sd_method_id),
    dataset_id INTEGER REFERENCES datasets(dataset_id),
    UNIQUE (target_model_id, sd_model_id, sd_method_id, dataset_id)
);
CREATE TABLE IF NOT EXISTS ld_performances (
    date TIMESTAMP,
    load INTEGER,
    end_to_end_latency FLOAT,
    input_tokens INTEGER,
    num_spec_tokens INTEGER,
    sd_setup_id INTEGER REFERENCES sd_setups(sd_setup_id)
);
CREATE TABLE IF NOT EXISTS sd_performances (
    date TIMESTAMP,
    sd_setup_id INTEGER REFERENCES sd_setups(sd_setup_id),
    mean_acceptance_length FLOAT,
    time_taken FLOAT,
    rate_at_1position FLOAT,
    rate_at_2position FLOAT,
    rate_at_3position FLOAT,
    rate_at_4position FLOAT,
    rate_at_5position FLOAT,
    input_tokens INTEGER
);
"""

_TABLES_WITH_RLS = [
    "models", "sd_methods", "datasets",
    "sd_setups", "ld_performances", "sd_performances",
]


_SEQUENCE_COLUMNS = [
    ("models",    "model_id"),
    ("sd_methods","sd_method_id"),
    ("datasets",  "dataset_id"),
    ("sd_setups", "sd_setup_id"),
]


class PostgresBackend(BaseDBBackend):
    _ph = "%s"

    def __init__(self, conn) -> None:
        self.conn = conn

    def _cursor(self):
        return self.conn.cursor()

    def _commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        try:
            self.conn.rollback()
        except Exception:
            pass

    def _ensure_sequences(self) -> None:
        for table, id_col in _SEQUENCE_COLUMNS:
            seq = f"{table}_{id_col}_seq"
            try:
                cur = self.conn.cursor()
                cur.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq}")
                cur.execute(
                    f"SELECT setval(%s, GREATEST(COALESCE((SELECT MAX({id_col}) FROM {table}), 1), 1))",
                    (seq,),
                )
                cur.execute(
                    f"ALTER TABLE {table} ALTER COLUMN {id_col} SET DEFAULT nextval(%s)",
                    (seq,),
                )
                self.conn.commit()
                logger.info(f"Sequence {seq} configured for {table}.{id_col}")
            except Exception as e:
                logger.error(f"Failed to configure sequence {seq}: {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def create_tables(self) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(_PG_CREATE_TABLES)
            for table in _TABLES_WITH_RLS:
                cur.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
                cur.execute(f"""
                    DO $$ BEGIN
                        CREATE POLICY public_read ON {table} FOR SELECT USING (true);
                    EXCEPTION WHEN duplicate_object THEN NULL;
                    END $$
                """)
            self.conn.commit()
            logger.info("Supabase tables created/verified.")
        except Exception as e:
            logger.warning(f"Supabase table setup had issues: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
        self._ensure_sequences()
