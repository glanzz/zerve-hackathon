"""sql_runner.py — thin DuckDB wrapper.

Functions
---------
run_sql_file(sql_path, params=None, output_parquet=None) -> pd.DataFrame
run_sql_string(sql, params=None, output_parquet=None) -> pd.DataFrame
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def _execute(sql: str, params: Optional[dict], output_parquet: Optional[str | Path]) -> pd.DataFrame:
    """Run *sql* in an in-process DuckDB connection and return a DataFrame."""
    conn = duckdb.connect()
    if params:
        for key, val in params.items():
            conn.execute(f"SET {key} = {repr(val)}")
    result: pd.DataFrame = conn.execute(sql).df()
    conn.close()
    if output_parquet is not None:
        _out = Path(output_parquet)
        _out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(_out, index=False)
    return result


def run_sql_file(
    sql_path: str | Path,
    params: Optional[dict] = None,
    output_parquet: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Read *sql_path* and execute it via DuckDB.

    Parameters
    ----------
    sql_path : path to .sql file
    params : optional dict of DuckDB SET key=value pairs
    output_parquet : if given, write result to this parquet path

    Returns
    -------
    pd.DataFrame with query results
    """
    _p = Path(sql_path)
    if not _p.exists():
        raise FileNotFoundError(f"SQL file not found: {_p}")
    _sql = _p.read_text(encoding="utf-8")
    return _execute(_sql, params, output_parquet)


def run_sql_string(
    sql: str,
    params: Optional[dict] = None,
    output_parquet: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Execute a raw SQL string via DuckDB.

    Parameters
    ----------
    sql : SQL query string
    params : optional dict of DuckDB SET key=value pairs
    output_parquet : if given, write result to this parquet path

    Returns
    -------
    pd.DataFrame with query results
    """
    return _execute(sql, params, output_parquet)
