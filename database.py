"""
PostgreSQL database layer for Supabase deployment.
Falls back to SQLite for local development.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

# Detect environment
USE_POSTGRES = False
try:
    import streamlit as st
    db_url = st.secrets.get("DATABASE_URL", "")
    if db_url:
        import psycopg2
        import psycopg2.extras
        USE_POSTGRES = True
except Exception:
    pass

SQLITE_PATH = Path("cat_prep.db")


# ─── Connection helpers ───────────────────────────────────────────────────────

def _get_pg_conn():
    import streamlit as st
    import psycopg2
    
    db_url = st.secrets["DATABASE_URL"]
    
    if "sslmode" not in db_url:
        db_url += "?sslmode=require"
    
    return psycopg2.connect(db_url)

def _get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_connection():
    if USE_POSTGRES:
        return _get_pg_conn()
    return _get_sqlite_conn()


# ─── Schema ───────────────────────────────────────────────────────────────────

def initialize_db():
    if USE_POSTGRES:
        conn = _get_pg_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS readiness_snapshots (
                id                SERIAL PRIMARY KEY,
                date              TEXT NOT NULL UNIQUE,
                readiness_score   REAL,
                accuracy_score    REAL,
                speed_score       REAL,
                coverage_score    REAL,
                consistency_score REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS readiness_snapshots (
                id                INTEGER SERIAL PRIMARY KEY,
                date              TEXT NOT NULL UNIQUE,
                readiness_score   REAL,
                accuracy_score    REAL,
                speed_score       REAL,
                coverage_score    REAL,
                consistency_score REAL
            )
        """)
        conn.commit()
        conn.close()
    else:
        # Original SQLite schema — unchanged
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                date                TEXT NOT NULL,
                section             TEXT NOT NULL,
                topic               TEXT NOT NULL,
                subtopic            TEXT,
                activity_type       TEXT DEFAULT 'practice',
                questions_attempted INTEGER DEFAULT 0,
                sets_attempted      INTEGER DEFAULT 0,
                correct_answers     INTEGER DEFAULT 0,
                correct_sets        INTEGER DEFAULT 0,
                time_taken_minutes  REAL DEFAULT 0,
                sentiment           TEXT DEFAULT 'medium',
                raw_input           TEXT,
                created_at          TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS readiness_snapshots (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                date              TEXT NOT NULL UNIQUE,
                readiness_score   REAL,
                accuracy_score    REAL,
                speed_score       REAL,
                coverage_score    REAL,
                consistency_score REAL
            )
        """)
        conn.commit()
        conn.close()


def insert_logs(entries: list[dict], raw_input: str = "") -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    inserted = 0

    if USE_POSTGRES:
        conn = _get_pg_conn()
        cursor = conn.cursor()
        for entry in entries:
            cursor.execute("""
                INSERT INTO logs (
                    date, section, topic, subtopic, activity_type,
                    questions_attempted, sets_attempted,
                    correct_answers, correct_sets,
                    time_taken_minutes, sentiment, raw_input
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                entry.get("date", today),
                entry.get("section", ""),
                entry.get("topic", ""),
                entry.get("subtopic", ""),
                entry.get("activity_type", "practice"),
                int(entry.get("questions_attempted", 0)),
                int(entry.get("sets_attempted", 0)),
                int(entry.get("correct_answers", 0)),
                int(entry.get("correct_sets", 0)),
                float(entry.get("time_taken_minutes", 0)),
                entry.get("sentiment", "medium"),
                raw_input,
            ))
            inserted += 1
        conn.commit()
        conn.close()
    else:
        # Original SQLite path
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        for entry in entries:
            cursor.execute("""
                INSERT INTO logs (
                    date, section, topic, subtopic, activity_type,
                    questions_attempted, sets_attempted,
                    correct_answers, correct_sets,
                    time_taken_minutes, sentiment, raw_input
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                entry.get("date", today),
                entry.get("section", ""),
                entry.get("topic", ""),
                entry.get("subtopic", ""),
                entry.get("activity_type", "practice"),
                int(entry.get("questions_attempted", 0)),
                int(entry.get("sets_attempted", 0)),
                int(entry.get("correct_answers", 0)),
                int(entry.get("correct_sets", 0)),
                float(entry.get("time_taken_minutes", 0)),
                entry.get("sentiment", "medium"),
                raw_input,
            ))
            inserted += 1
        conn.commit()
        conn.close()

    return inserted


def fetch_all_logs(
    start_date=None, end_date=None,
    section=None, topic=None,
) -> pd.DataFrame:
    query = "SELECT * FROM logs WHERE 1=1"
    params = []

    if start_date:
        query += " AND date >= %s" if USE_POSTGRES else " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= %s" if USE_POSTGRES else " AND date <= ?"
        params.append(end_date)
    if section and section != "All":
        query += " AND section = %s" if USE_POSTGRES else " AND section = ?"
        params.append(section)
    if topic and topic != "All":
        query += " AND topic = %s" if USE_POSTGRES else " AND topic = ?"
        params.append(topic)

    query += " ORDER BY date DESC, id DESC"

    if USE_POSTGRES:
        conn = _get_pg_conn()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    else:
        conn = _get_sqlite_conn()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

    return df


def save_readiness_snapshot(date: str, scores: dict):
    if USE_POSTGRES:
        conn = _get_pg_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO readiness_snapshots
                (date, readiness_score, accuracy_score, speed_score,
                 coverage_score, consistency_score)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (date) DO UPDATE SET
                readiness_score   = EXCLUDED.readiness_score,
                accuracy_score    = EXCLUDED.accuracy_score,
                speed_score       = EXCLUDED.speed_score,
                coverage_score    = EXCLUDED.coverage_score,
                consistency_score = EXCLUDED.consistency_score
        """, (
            date,
            scores.get("readiness_score", 0),
            scores.get("accuracy_score", 0),
            scores.get("speed_score", 0),
            scores.get("coverage_score", 0),
            scores.get("consistency_score", 0),
        ))
        conn.commit()
        conn.close()
    else:
        conn = _get_sqlite_conn()
        conn.execute("""
            INSERT OR REPLACE INTO readiness_snapshots
                (date, readiness_score, accuracy_score, speed_score,
                 coverage_score, consistency_score)
            VALUES (?,?,?,?,?,?)
        """, (
            date,
            scores.get("readiness_score", 0),
            scores.get("accuracy_score", 0),
            scores.get("speed_score", 0),
            scores.get("coverage_score", 0),
            scores.get("consistency_score", 0),
        ))
        conn.commit()
        conn.close()


def fetch_readiness_history() -> pd.DataFrame:
    query = "SELECT * FROM readiness_snapshots ORDER BY date ASC"
    if USE_POSTGRES:
        conn = _get_pg_conn()
    else:
        conn = _get_sqlite_conn()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def delete_log(log_id: int):
    ph = "%s" if USE_POSTGRES else "?"
    if USE_POSTGRES:
        conn = _get_pg_conn()
    else:
        conn = _get_sqlite_conn()
    conn.execute(f"DELETE FROM logs WHERE id = {ph}", (log_id,))
    conn.commit()
    conn.close()
