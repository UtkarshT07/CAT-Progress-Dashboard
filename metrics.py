"""
Metrics and scoring engine.
Computes accuracy, speed, difficulty scores and topic classifications.
"""

import pandas as pd
import numpy as np
from cat_structure import (
    CAT_STRUCTURE, SENTIMENT_PENALTY, SPEED_BENCHMARKS, get_all_topics
)


# ─── Per-row helpers ────────────────────────────────────────────────────────

def compute_accuracy(row: pd.Series) -> float:
    """Accuracy as a fraction 0–1."""
    section = row.get("section", "QA")
    if section == "DILR":
        attempted = row.get("sets_attempted", 0)
        correct = row.get("correct_sets", 0)
    else:
        attempted = row.get("questions_attempted", 0)
        correct = row.get("correct_answers", 0)

    if attempted <= 0:
        return 0.0
    return min(1.0, correct / attempted)


def compute_time_per_unit(row: pd.Series) -> float:
    """Time per question or per set, in seconds."""
    section = row.get("section", "QA")
    minutes = row.get("time_taken_minutes", 0) or 0

    if section == "DILR":
        units = row.get("sets_attempted", 0) or 0
    else:
        units = row.get("questions_attempted", 0) or 0

    if units <= 0:
        return 0.0
    return (minutes * 60) / units


def compute_difficulty_score(row: pd.Series) -> float:
    """
    Higher score → more difficult / needs more work.
    score = (100 - accuracy%) + speed_penalty + sentiment_penalty
    """
    accuracy_pct = compute_accuracy(row) * 100
    time_per_unit = compute_time_per_unit(row)
    section = row.get("section", "QA")
    sentiment = row.get("sentiment", "medium")

    # Speed penalty: how much over the "good" benchmark
    benchmark = SPEED_BENCHMARKS.get(section, {}).get("good", 90)
    speed_penalty = max(0, (time_per_unit - benchmark) / benchmark * 20)

    penalty = SENTIMENT_PENALTY.get(sentiment, 5)

    score = (100 - accuracy_pct) + speed_penalty + penalty
    return round(min(score, 130), 2)  # cap at 130


# ─── Aggregated topic-level stats ────────────────────────────────────────────

def compute_topic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (section, topic) with aggregated metrics.
    Includes classification: Strong / Medium / Weak / Untouched.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["accuracy"] = df.apply(compute_accuracy, axis=1)
    df["time_per_unit"] = df.apply(compute_time_per_unit, axis=1)
    df["difficulty_score"] = df.apply(compute_difficulty_score, axis=1)

    # DILR uses sets; others use questions
    df["units_attempted"] = df.apply(
        lambda r: r["sets_attempted"] if r["section"] == "DILR"
        else r["questions_attempted"], axis=1
    )
    df["units_correct"] = df.apply(
        lambda r: r["correct_sets"] if r["section"] == "DILR"
        else r["correct_answers"], axis=1
    )

    grouped = df.groupby(["section", "topic"]).agg(
        total_units_attempted=("units_attempted", "sum"),
        total_units_correct=("units_correct", "sum"),
        total_time_minutes=("time_taken_minutes", "sum"),
        avg_difficulty=("difficulty_score", "mean"),
        sessions=("id", "count"),
        last_date=("date", "max"),
    ).reset_index()

    # Recompute accuracy from totals
    grouped["accuracy"] = (
        grouped["total_units_correct"] / grouped["total_units_attempted"]
    ).fillna(0).clip(0, 1)

    # Average time per unit (seconds)
    grouped["avg_time_per_unit"] = grouped.apply(
        lambda r: (r["total_time_minutes"] * 60 / r["total_units_attempted"])
        if r["total_units_attempted"] > 0 else 0, axis=1
    )

    grouped["classification"] = grouped.apply(classify_topic, axis=1)

    return grouped


def classify_topic(row: pd.Series) -> str:
    """Strong / Medium / Weak based on accuracy and speed."""
    acc = row["accuracy"]
    section = row.get("section", "QA")
    time_per_unit = row.get("avg_time_per_unit", 0)
    good_speed = SPEED_BENCHMARKS.get(section, {}).get("good", 90)

    if acc >= 0.80 and time_per_unit <= good_speed * 1.2:
        return "Strong"
    elif acc >= 0.60:
        return "Medium"
    else:
        return "Weak"


def get_untouched_topics(df: pd.DataFrame) -> list[dict]:
    """Return topics that have no log entries at all."""
    touched = set()
    if not df.empty:
        for _, row in df.iterrows():
            touched.add((row["section"], row["topic"]))

    untouched = []
    for section, topic_map in CAT_STRUCTURE.items():
        for topic in topic_map:
            if (section, topic) not in touched:
                untouched.append({"section": section, "topic": topic})
    return untouched


# ─── Overview stats ──────────────────────────────────────────────────────────

def compute_overview(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_questions": 0,
            "total_sets": 0,
            "total_correct_answers": 0,
            "total_correct_sets": 0,
            "overall_accuracy": 0.0,
            "total_time_hours": 0.0,
            "total_sessions": 0,
        }

    return {
        "total_questions": int(df["questions_attempted"].sum()),
        "total_sets": int(df["sets_attempted"].sum()),
        "total_correct_answers": int(df["correct_answers"].sum()),
        "total_correct_sets": int(df["correct_sets"].sum()),
        "overall_accuracy": float(
            df.apply(compute_accuracy, axis=1).mean()
        ),
        "total_time_hours": float(df["time_taken_minutes"].sum() / 60),
        "total_sessions": len(df),
    }


def compute_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Daily aggregated accuracy and time for trend charts."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["accuracy"] = df.apply(compute_accuracy, axis=1)
    df["units"] = df.apply(
        lambda r: r["sets_attempted"] if r["section"] == "DILR"
        else r["questions_attempted"], axis=1
    )

    daily = df.groupby("date").agg(
        avg_accuracy=("accuracy", "mean"),
        total_time=("time_taken_minutes", "sum"),
        total_units=("units", "sum"),
        sessions=("id", "count"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    return daily


def get_weakest_topics(topic_stats: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if topic_stats.empty:
        return pd.DataFrame()
    return (
        topic_stats[topic_stats["classification"] != "Untouched"]
        .nlargest(n, "avg_difficulty")
    )


def get_section_stats(df: pd.DataFrame) -> dict:
    """Per-section summary."""
    stats = {}
    for section in ["QA", "DILR", "VARC"]:
        sub = df[df["section"] == section]
        if sub.empty:
            stats[section] = None
            continue

        if section == "DILR":
            attempted = int(sub["sets_attempted"].sum())
            correct = int(sub["correct_sets"].sum())
        else:
            attempted = int(sub["questions_attempted"].sum())
            correct = int(sub["correct_answers"].sum())

        accuracy = correct / attempted if attempted > 0 else 0

        stats[section] = {
            "attempted": attempted,
            "correct": correct,
            "accuracy": accuracy,
            "time_hours": float(sub["time_taken_minutes"].sum() / 60),
            "sessions": len(sub),
        }
    return stats
