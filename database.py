import os
import sqlite3
from datetime import datetime, timezone


DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "neuroshield.db")


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                prediction TEXT NOT NULL,
                safe_probability REAL NOT NULL,
                injection_probability REAL NOT NULL,
                threshold REAL NOT NULL,
                action TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def create_scan(
    prompt: str,
    prediction: str,
    safe_probability: float,
    injection_probability: float,
    threshold: float,
    action: str,
    latency_ms: float,
):
    created_at = datetime.now(timezone.utc).isoformat()
    connection = get_connection()
    try:
        cursor = connection.execute(
            """
            INSERT INTO scans (
                prompt, prediction, safe_probability,
                injection_probability, threshold, action,
                latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prompt, prediction, safe_probability,
                injection_probability, threshold, action,
                latency_ms, created_at,
            ),
        )
        connection.commit()
        return {
            "id": cursor.lastrowid,
            "prompt": prompt,
            "prediction": prediction,
            "safe_probability": safe_probability,
            "injection_probability": injection_probability,
            "threshold": threshold,
            "action": action,
            "latency_ms": latency_ms,
            "created_at": created_at,
        }
    finally:
        connection.close()


def get_scans(limit: int = 50):
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, prompt, prediction, safe_probability,
                   injection_probability, threshold, action,
                   latency_ms, created_at
            FROM scans
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def clear_scans():
    connection = get_connection()
    try:
        connection.execute("DELETE FROM scans")
        connection.commit()
    finally:
        connection.close()