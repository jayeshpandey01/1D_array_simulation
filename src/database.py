"""
SQLite storage for signals and analysis results.

Default path: data/signals.db

Connects CSV pipeline ↔ database ↔ GUI.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from src.pipeline import analyze_one
from src.signal_io import ground_truth_from_row, load_signal_csv

DEFAULT_DB_PATH = os.path.join("data", "signals.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> str:
    """Create tables if they do not exist."""
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wave_type TEXT,
                frequency REAL,
                amplitude REAL,
                phase REAL,
                snr_db REAL,
                signal_json TEXT NOT NULL,
                source_file TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                wave_type_detected TEXT,
                freq_detected REAL,
                amp_detected REAL,
                phase_detected REAL,
                freq_error_hz REAL,
                freq_error_pct REAL,
                amp_error REAL,
                phase_error_rad REAL,
                report_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            );
            """
        )
    return os.path.abspath(db_path)


def import_csv(
    csv_path: str,
    db_path: str = DEFAULT_DB_PATH,
    source_label: Optional[str] = None,
) -> int:
    """Import all rows from a labeled CSV into the database. Returns rows inserted."""
    init_db(db_path)
    signals, meta = load_signal_csv(csv_path)
    label = source_label or os.path.basename(csv_path)
    count = 0

    with connect(db_path) as conn:
        for i in range(signals.shape[0]):
            gt = ground_truth_from_row(meta.iloc[i]) if meta is not None else {}
            gt = gt or {}
            conn.execute(
                """
                INSERT INTO signals (
                    wave_type, frequency, amplitude, phase, snr_db,
                    signal_json, source_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gt.get("wave_type") or gt.get("label_name"),
                    gt.get("frequency"),
                    gt.get("amplitude"),
                    gt.get("phase"),
                    gt.get("snr_db"),
                    json.dumps(signals[i].tolist()),
                    label,
                    _utc_now(),
                ),
            )
            count += 1
    return count


def list_signals(db_path: str = DEFAULT_DB_PATH, limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent signal records (without full waveform)."""
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, wave_type, frequency, amplitude, phase, source_file, created_at
            FROM signals ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_signal(
    signal_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Load one signal array and ground-truth metadata by database id."""
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    if row is None:
        raise KeyError(f"No signal with id={signal_id}")

    sig = np.array(json.loads(row["signal_json"]), dtype=np.float64)
    gt = {
        "wave_type": row["wave_type"],
        "frequency": row["frequency"],
        "amplitude": row["amplitude"],
        "phase": row["phase"],
        "snr_db": row["snr_db"],
    }
    gt = {k: v for k, v in gt.items() if v is not None}
    return sig, (gt if gt else None)


def save_analysis(
    signal_id: int,
    result_dict: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Persist an analysis result linked to signal_id."""
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO analyses (
                signal_id, wave_type_detected, freq_detected, amp_detected,
                phase_detected, freq_error_hz, freq_error_pct, amp_error,
                phase_error_rad, report_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                result_dict.get("wave_type") or result_dict.get("wave_type_detected"),
                result_dict.get("frequency_hz") or result_dict.get("freq_detected"),
                result_dict.get("amplitude") or result_dict.get("amp_detected"),
                result_dict.get("phase_rad") or result_dict.get("phase_detected"),
                result_dict.get("frequency_error_hz") or result_dict.get("freq_error_hz"),
                result_dict.get("frequency_error_pct") or result_dict.get("freq_error_pct"),
                result_dict.get("amplitude_error") or result_dict.get("amp_error"),
                result_dict.get("phase_error_rad"),
                json.dumps(result_dict, default=str),
                _utc_now(),
            ),
        )
        return int(cur.lastrowid)


def analyze_and_store(signal_id: int, db_path: str = DEFAULT_DB_PATH):
    """Run pipeline analysis on a DB signal and save the result."""
    sig, gt = get_signal(signal_id, db_path)
    result = analyze_one(sig, ground_truth=gt)
    save_analysis(signal_id, result.to_dict(), db_path)
    return result


def export_to_csv(
    db_path: str = DEFAULT_DB_PATH,
    output_path: str = "datasets/from_database.csv",
) -> str:
    """Export all signals from DB to a pipeline-compatible CSV."""
    init_db(db_path)
    rows_out = []
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()

    for row in rows:
        sig = np.array(json.loads(row["signal_json"]), dtype=np.float64)
        meta = {
            "wave_type": row["wave_type"],
            "frequency": row["frequency"],
            "amplitude": row["amplitude"],
            "phase": row["phase"],
            "snr_db": row["snr_db"],
            "db_id": row["id"],
        }
        rows_out.append((sig, meta))

    if not rows_out:
        raise ValueError("Database has no signals to export.")

    signals = np.vstack([r[0] for r in rows_out])
    meta_list = [r[1] for r in rows_out]
    from src.signal_io import signals_to_dataframe

    df = signals_to_dataframe(signals, meta_list)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
