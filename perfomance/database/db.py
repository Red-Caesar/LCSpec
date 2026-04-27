import logging
from typing import List, Optional, Tuple

from perfomance.database.backends.base import BaseDBBackend

logger = logging.getLogger(__name__)


class DatabaseClient:
    def __init__(
        self,
        local: BaseDBBackend,
        remote: Optional[BaseDBBackend] = None,
    ) -> None:
        self.local = local
        self.remote = remote

    def create_tables(self) -> None:
        self.local.create_tables()
        if self.remote:
            self._try_remote(self.remote.create_tables)

    def resolve_sd_setup(self, data: dict) -> Tuple[int, Optional[int]]:
        local_id = self.local.resolve_sd_setup(data)
        remote_id = (
            self._try_remote(self.remote.resolve_sd_setup, data)
            if self.remote
            else None
        )
        return local_id, remote_id

    def insert_load_test_performance(
        self,
        local_id: int,
        load: int,
        end_to_end_latency: float,
        input_tokens: int,
        num_spec_tokens: int,
        date: str,
        remote_id: Optional[int] = None,
    ) -> None:
        self.local.insert_load_test_performance(
            local_id, load, end_to_end_latency, input_tokens, num_spec_tokens, date
        )
        if self.remote and remote_id is not None:
            self._try_remote(
                self.remote.insert_load_test_performance,
                remote_id, load, end_to_end_latency, input_tokens, num_spec_tokens, date,
            )

    def insert_sd_performance(
        self,
        local_id: int,
        mean_acceptance_length: float,
        date: str,
        time_taken: float,
        acceptance_rates: List[float],
        input_tokens: int,
        remote_id: Optional[int] = None,
    ) -> None:
        self.local.insert_sd_performance(
            local_id, mean_acceptance_length, date, time_taken, acceptance_rates, input_tokens
        )
        if self.remote and remote_id is not None:
            self._try_remote(
                self.remote.insert_sd_performance,
                remote_id, mean_acceptance_length, date, time_taken, acceptance_rates, input_tokens,
            )

    def _try_remote(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Remote DB write failed: {e}")
            self.remote.rollback()
            return None
