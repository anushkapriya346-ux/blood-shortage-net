"""SQLite persistence layer for the Predictive Blood Shortage Network.

Provides schema initialization (`init_db`), table loading (`load_table`), and
table saving (`save_table`) helpers backed by a local SQLite database.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "blood_network.db"

VALID_TABLES = {
    "facilities",
    "inventory_units",
    "daily_consumption",
    "emergency_events",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facilities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('hospital', 'blood_bank')),
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    region TEXT NOT NULL,
    capacity_factor REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory_units (
    id TEXT PRIMARY KEY,
    facility_id TEXT NOT NULL,
    blood_group TEXT NOT NULL,
    component TEXT NOT NULL CHECK(component IN ('RBC', 'Platelets')),
    collected_date TEXT NOT NULL,
    expiry_date TEXT NOT NULL,
    FOREIGN KEY (facility_id) REFERENCES facilities(id)
);

CREATE TABLE IF NOT EXISTS daily_consumption (
    date TEXT NOT NULL,
    facility_id TEXT NOT NULL,
    blood_group TEXT NOT NULL,
    component TEXT NOT NULL CHECK(component IN ('RBC', 'Platelets')),
    units_used INTEGER NOT NULL,
    PRIMARY KEY (date, facility_id, blood_group, component),
    FOREIGN KEY (facility_id) REFERENCES facilities(id)
);

CREATE TABLE IF NOT EXISTS emergency_events (
    date TEXT NOT NULL,
    facility_id TEXT NOT NULL,
    type TEXT NOT NULL,
    multiplier REAL NOT NULL,
    FOREIGN KEY (facility_id) REFERENCES facilities(id)
);

CREATE INDEX IF NOT EXISTS idx_inv_fac_group
    ON inventory_units(facility_id, blood_group, component);

CREATE INDEX IF NOT EXISTS idx_cons_fac_group_date
    ON daily_consumption(facility_id, blood_group, date);

CREATE INDEX IF NOT EXISTS idx_emerg_fac_date
    ON emergency_events(facility_id, date);
"""


def get_db_path(db_path: Optional[str | Path] = None) -> Path:
    """Resolve the SQLite database path."""
    if db_path is not None:
        return Path(db_path)
    env_path = os.environ.get("BLOOD_NET_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def get_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str | Path] = None) -> Path:
    """Initialize the SQLite database schema if tables do not exist."""
    path = get_db_path(db_path)
    with get_connection(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    return path


def _validate_table_name(name: str) -> None:
    if name not in VALID_TABLES:
        raise ValueError(
            f"Unknown table '{name}'. Expected one of {sorted(VALID_TABLES)}."
        )


def load_table(name: str, db_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load an entire SQLite table into a pandas DataFrame."""
    _validate_table_name(name)
    path = init_db(db_path)
    with get_connection(path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {name}", conn)


def save_table(
    df: pd.DataFrame,
    name: str,
    db_path: Optional[str | Path] = None,
    append: bool = False,
) -> int:
    """Save a DataFrame into a SQLite table, replacing existing rows by default."""
    _validate_table_name(name)
    path = init_db(db_path)
    with get_connection(path) as conn:
        if not append:
            conn.execute(f"DELETE FROM {name};")
        if df is not None and not df.empty:
            df.to_sql(name, conn, if_exists="append", index=False)
        conn.commit()
    return 0 if df is None else len(df)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_blood.db"
        init_db(test_db)
        sample_fac = pd.DataFrame(
            [
                {
                    "id": "FAC01",
                    "name": "Test Medical College",
                    "type": "hospital",
                    "lat": 25.61,
                    "lon": 85.14,
                    "region": "Central Patna",
                    "capacity_factor": 1.5,
                }
            ]
        )
        rows = save_table(sample_fac, "facilities", db_path=test_db)
        loaded = load_table("facilities", db_path=test_db)
        empty_inv = load_table("inventory_units", db_path=test_db)
        assert rows == 1
        assert len(loaded) == 1
        assert loaded.loc[0, "name"] == "Test Medical College"
        assert empty_inv.empty
        print(f"db.py self-test OK: saved={rows}, loaded={len(loaded)}, empty_inv={len(empty_inv)}")
