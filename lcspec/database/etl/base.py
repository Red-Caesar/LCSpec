from pathlib import Path
from typing import Any, Dict, Tuple

class ETLBase:
    def __init__(self, db_name: str):
        self.db_name = db_name

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
