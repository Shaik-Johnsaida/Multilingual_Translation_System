"""
Local SQLite Storage Engine for Translation History.
"""

import os
import sqlite3
import time
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "history.db")


def init_db():
    """Initializes SQLite database and tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                mode TEXT,
                source_lang TEXT,
                target_lang TEXT,
                source_text TEXT,
                translated_text TEXT,
                model_used TEXT,
                processing_time_sec REAL
            )
        """)
        conn.commit()


def record_history(
    mode: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    translated_text: str,
    model_used: str,
    processing_time_sec: float
):
    """Records a translation entry in SQLite history."""
    init_db()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO translation_history 
            (timestamp, mode, source_lang, target_lang, source_text, translated_text, model_used, processing_time_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, mode, source_lang, target_lang, source_text[:500], translated_text[:500], model_used, processing_time_sec))
        conn.commit()


def get_translation_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent translation entries from SQLite."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM translation_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
