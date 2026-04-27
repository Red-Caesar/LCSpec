import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def connect_supabase() -> Optional[object]:
    """Connect to Supabase via SUPABASE_DB_URL; return connection or None."""
    try:
        import psycopg2
        from dotenv import load_dotenv
    except ImportError:
        logger.warning(
            "psycopg2-binary or python-dotenv not installed; skipping Supabase."
        )
        return None

    load_dotenv(dotenv_path=_ENV_PATH)
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        logger.info("SUPABASE_DB_URL not set; skipping Supabase connection.")
        return None

    try:
        conn = psycopg2.connect(db_url)
        logger.info("Connected to Supabase successfully.")
        return conn
    except Exception as e:
        logger.warning(f"Failed to connect to Supabase: {e}")
        return None
