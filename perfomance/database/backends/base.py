from abc import ABC, abstractmethod
from typing import List


class BaseDBBackend(ABC):
    _ph: str

    @abstractmethod
    def _cursor(self):
        ...

    @abstractmethod
    def _commit(self) -> None:
        ...

    def rollback(self) -> None:
        pass

    @abstractmethod
    def create_tables(self) -> None:
        ...

    def _get_or_insert_dimension(
        self, table: str, id_col: str, name_col: str, value: str
    ) -> int:
        ph = self._ph
        cur = self._cursor()
        cur.execute(
            f"INSERT INTO {table} ({name_col}) VALUES ({ph})"
            f" ON CONFLICT ({name_col}) DO NOTHING RETURNING {id_col}",
            (value,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            f"SELECT {id_col} FROM {table} WHERE {name_col} = {ph}", (value,)
        )
        return cur.fetchone()[0]

    def _get_or_insert_sd_setup(
        self,
        target_model_id: int,
        sd_model_id: int,
        sd_method_id: int,
        dataset_id: int,
    ) -> int:
        ph = self._ph
        cur = self._cursor()
        cur.execute(
            "INSERT INTO sd_setups"
            " (target_model_id, sd_model_id, sd_method_id, dataset_id)"
            f" VALUES ({ph}, {ph}, {ph}, {ph})"
            " ON CONFLICT (target_model_id, sd_model_id, sd_method_id, dataset_id)"
            " DO NOTHING RETURNING sd_setup_id",
            (target_model_id, sd_model_id, sd_method_id, dataset_id),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            f"SELECT sd_setup_id FROM sd_setups"
            f" WHERE target_model_id={ph} AND sd_model_id={ph}"
            f" AND sd_method_id={ph} AND dataset_id={ph}",
            (target_model_id, sd_model_id, sd_method_id, dataset_id),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                f"Failed to resolve sd_setups for "
                f"({target_model_id}, {sd_model_id}, {sd_method_id}, {dataset_id})"
            )
        return row[0]

    def resolve_sd_setup(self, data: dict) -> int:
        """Get or create all dimension records; return sd_setup_id for this DB."""
        target_id = self._get_or_insert_dimension(
            "models", "model_id", "model_name", data["target_model"]
        )
        sd_model_id = self._get_or_insert_dimension(
            "models", "model_id", "model_name", data["sd_model"]
        )
        sd_method_id = self._get_or_insert_dimension(
            "sd_methods", "sd_method_id", "sd_method_type", data["sd_method"]
        )
        dataset_id = self._get_or_insert_dimension(
            "datasets", "dataset_id", "dataset_type", data["dataset_type"]
        )
        sd_setup_id = self._get_or_insert_sd_setup(
            target_id, sd_model_id, sd_method_id, dataset_id
        )
        self._commit()
        return sd_setup_id

    def insert_load_test_performance(
        self,
        sd_setup_id: int,
        load: int,
        end_to_end_latency: float,
        input_tokens: int,
        num_spec_tokens: int,
        date: str,
    ) -> None:
        ph = self._ph
        self._cursor().execute(
            "INSERT INTO ld_performances"
            " (sd_setup_id, load, end_to_end_latency, input_tokens, num_spec_tokens, date)"
            f" VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})",
            (sd_setup_id, load, end_to_end_latency, input_tokens, num_spec_tokens, date),
        )
        self._commit()

    def insert_sd_performance(
        self,
        sd_setup_id: int,
        mean_acceptance_length: float,
        date: str,
        time_taken: float,
        acceptance_rates: List[float],
        input_tokens: int,
    ) -> None:
        if len(acceptance_rates) < 5:
            acceptance_rates = acceptance_rates + [0.0] * (5 - len(acceptance_rates))
        ar_1, ar_2, ar_3, ar_4, ar_5 = acceptance_rates
        ph = self._ph
        self._cursor().execute(
            "INSERT INTO sd_performances"
            " (date, sd_setup_id, mean_acceptance_length, time_taken,"
            " rate_at_1position, rate_at_2position, rate_at_3position,"
            " rate_at_4position, rate_at_5position, input_tokens)"
            f" VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})",
            (
                date, sd_setup_id, mean_acceptance_length, time_taken,
                ar_1, ar_2, ar_3, ar_4, ar_5, input_tokens,
            ),
        )
        self._commit()
