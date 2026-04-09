"""
Readiness estimation engine.
Computes a 0–100 score and projects time-to-readiness.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from metrics import compute_topic_stats, get_untouched_topics
from cat_structure import CAT_STRUCTURE, SPEED_BENCHMARKS


TARGET_SCORE = 85.0

READINESS_LEVELS = [
    (85, "🏆 Exam Ready"),
    (70, "🚀 Competitive"),
    (40, "📈 Developing"),
    (0,  "🌱 Early Stage"),
]


def get_readiness_level(score: float) -> str:
    for threshold, label in READINESS_LEVELS:
        if score >= threshold:
            return label
    return "🌱 Early Stage"


def compute_readiness_scores(df: pd.DataFrame) -> dict:
    """
    Compute component scores and overall readiness (0–100).

    Components:
      - accuracy_score   (0–40): weighted average accuracy
      - speed_score      (0–20): speed vs benchmark
      - coverage_score   (0–25): % topics touched with ≥60% accuracy
      - consistency_score(0–15): study streak / regularity
    """
    if df.empty:
        return _zero_scores()

    from metrics import compute_accuracy, compute_time_per_unit

    df = df.copy()
    df["accuracy"] = df.apply(compute_accuracy, axis=1)
    df["time_per_unit"] = df.apply(compute_time_per_unit, axis=1)

    # ── Accuracy score (0–40) ────────────────────────────────
    avg_acc = df["accuracy"].mean()
    accuracy_score = round(avg_acc * 40, 2)

    # ── Speed score (0–20) ───────────────────────────────────
    speed_scores = []
    for section in ["QA", "DILR", "VARC"]:
        sub = df[df["section"] == section]
        if sub.empty:
            continue
        benchmark = SPEED_BENCHMARKS[section]["good"]
        avg_time = sub["time_per_unit"].mean()
        if avg_time <= 0:
            continue
        # 1.0 if at or below benchmark, decays as time increases
        ratio = benchmark / avg_time
        speed_scores.append(min(1.0, ratio))

    speed_score = round((np.mean(speed_scores) if speed_scores else 0) * 20, 2)

    # ── Coverage score (0–25) ────────────────────────────────
    topic_stats = compute_topic_stats(df)
    total_topics = sum(len(v) for v in CAT_STRUCTURE.values())

    if not topic_stats.empty:
        covered = topic_stats[topic_stats["accuracy"] >= 0.60]
        coverage_ratio = len(covered) / total_topics
    else:
        coverage_ratio = 0.0

    coverage_score = round(coverage_ratio * 25, 2)

    # ── Consistency score (0–15) ──────────────────────────────
    if "date" in df.columns and not df.empty:
        dates = pd.to_datetime(df["date"]).dt.date.unique()
        dates = sorted(dates)
        if len(dates) >= 2:
            date_range = (dates[-1] - dates[0]).days + 1
            active_days = len(dates)
            regularity = active_days / max(date_range, 1)
        else:
            regularity = 1.0 if len(dates) == 1 else 0.0
        consistency_score = round(min(1.0, regularity) * 15, 2)
    else:
        consistency_score = 0.0

    readiness_score = round(
        accuracy_score + speed_score + coverage_score + consistency_score, 2
    )
    readiness_score = min(100.0, readiness_score)

    return {
        "readiness_score": readiness_score,
        "accuracy_score": accuracy_score,
        "speed_score": speed_score,
        "coverage_score": coverage_score,
        "consistency_score": consistency_score,
        "level": get_readiness_level(readiness_score),
    }


def _zero_scores() -> dict:
    return {
        "readiness_score": 0.0,
        "accuracy_score": 0.0,
        "speed_score": 0.0,
        "coverage_score": 0.0,
        "consistency_score": 0.0,
        "level": "🌱 Early Stage",
    }


def estimate_time_to_readiness(
    readiness_history: pd.DataFrame,
    current_score: float,
) -> dict:
    """
    Linear regression on past readiness scores to project weeks to target.
    """
    result = {
        "weeks_min": None,
        "weeks_max": None,
        "weekly_rate": None,
        "projection_dates": [],
        "projection_scores": [],
    }

    if readiness_history.empty or len(readiness_history) < 2:
        # No history — use a default conservative estimate
        remaining = max(0, TARGET_SCORE - current_score)
        result["weeks_min"] = round(remaining / 3.5)
        result["weeks_max"] = round(remaining / 2.0)
        return result

    hist = readiness_history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")

    # Convert dates to days-from-start for regression
    start = hist["date"].min()
    hist["day_num"] = (hist["date"] - start).dt.days

    x = hist["day_num"].values
    y = hist["readiness_score"].values

    if len(x) < 2 or x.max() == 0:
        return result

    # Linear fit
    coeffs = np.polyfit(x, y, 1)  # slope (per day), intercept
    slope_per_day = coeffs[0]
    slope_per_week = slope_per_day * 7

    result["weekly_rate"] = round(slope_per_week, 2)

    if slope_per_week > 0:
        remaining = max(0, TARGET_SCORE - current_score)
        weeks_central = remaining / slope_per_week
        result["weeks_min"] = max(1, round(weeks_central * 0.75))
        result["weeks_max"] = max(2, round(weeks_central * 1.25))

        # Build projection curve (next 16 weeks)
        today = datetime.now().date()
        days_from_start = (today - start.date()).days
        for w in range(17):
            future_day = days_from_start + w * 7
            projected = coeffs[0] * future_day + coeffs[1]
            result["projection_dates"].append(
                (today + timedelta(weeks=w)).isoformat()
            )
            result["projection_scores"].append(round(min(100, max(0, projected)), 2))

    return result
