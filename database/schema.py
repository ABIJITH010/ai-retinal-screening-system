from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


logger = logging.getLogger(__name__)

DB_PATH = Path("database") / "retinal_screening.db"


def init_db(db_path: Path = DB_PATH) -> None:
    """
    Initialize SQLite database with required tables if they do not exist.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity_level INTEGER,
                risk_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
            """
        )

        conn.commit()

    logger.info("SQLite database initialized at %s", db_path)


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """
    Context manager that yields a SQLite connection with basic error handling.
    """
    try:
        conn = sqlite3.connect(db_path)
        yield conn
        conn.commit()
    except Exception as e:
        logger.exception("Database error: %s", str(e))
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_screening(
    prediction: str,
    confidence: float,
    severity_level: int,
    risk_level: str,
    patient_id: Optional[int] = None,
    db_path: Path = DB_PATH,
) -> int:
    """
    Insert a screening row and return its new ID.
    """
    init_db(db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO screenings (
                patient_id,
                prediction,
                confidence,
                severity_level,
                risk_level
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (patient_id, prediction, float(confidence), int(severity_level), risk_level),
        )
        screening_id = int(cursor.lastrowid)

    logger.info(
        "Stored screening in DB: id=%d prediction=%s confidence=%.4f severity=%d risk=%s",
        screening_id,
        prediction,
        confidence,
        severity_level,
        risk_level,
    )
    return screening_id

