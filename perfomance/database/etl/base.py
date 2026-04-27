from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from perfomance.database.db import DatabaseClient


class ETLBase:
    def __init__(self, db: "DatabaseClient"):
        self.db = db

    def _extract(self, file_path: Path | str) -> Any:
        raise NotImplementedError("Subclasses should implement this method.")

    def _transform(self, data: Any) -> Dict[Any, Any]:
        raise NotImplementedError("Subclasses should implement this method.")

    def _load(self, data: Dict[Any, Any]) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, file_path: Path | str) -> None:
        data = self._extract(file_path)
        transformed_data = self._transform(data)
        self._load(transformed_data)
