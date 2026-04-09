"""
AI suggestion engine using OpenAI.
Generates personalized study plans and insights.
"""

import json
from openai import OpenAI
import pandas as pd
from metrics import compute_topic_stats, get_untouched_topics
from readiness import get_readiness_level


def generate_suggestions(
    api_key: str,
    df: pd.DataFrame,
    readiness: dict,
    time_estimate: dict,
) -> dict:
    """
    Generate AI-powered suggestions.
    Returns dict with keys: insight, daily_plan, priorities, avoid, bottleneck.
    """
    if df.empty:
        return _default_suggestions()

    topic_stats = compute_topic_stats(df)
    untouched = get_untouched_topics(df)

    # Build a compact context for the LLM
    weak_topics = []
    medium_topics = []
    strong_topics = []

    if not topic_stats.empty:
        for _, row in topic_stats.iterrows():
            entry = f"{row['section']} → {row['topic']} ({row['accuracy']*100:.0f}% accuracy)"
            if row["classification"] == "Weak":
                weak_topics.append(entry)
            elif row["classification"] == "Medium":
                medium_topics.append(entry)
            else:
                strong_topics.append(entry)

    untouched_list = [
        f"{t['section']} → {t['topic']}" for t in untouched[:8]
    ]

    context = {
        "readiness_score": readiness.get("readiness_score", 0),
        "readiness_level": readiness.get("level", "Unknown"),
        "weeks_to_readiness": f"{time_estimate.get('weeks_min', '?')}–{time_estimate.get('weeks_max', '?')}",
        "weekly_improvement_rate": time_estimate.get("weekly_rate"),
        "weak_topics": weak_topics[:6],
        "medium_topics": medium_topics[:6],
        "strong_topics": strong_topics[:4],
        "untouched_topics": untouched_list,
        "accuracy_score": readiness.get("accuracy_score", 0),
        "speed_score": readiness.get("speed_score", 0),
        "coverage_score": readiness.get("coverage_score", 0),
        "consistency_score": readiness.get("consistency_score", 0),
    }

    system_prompt = """You are a world-class CAT (Common Admission Test) mentor.
Analyze the student's preparation data and provide concise, actionable guidance.

Respond in this exact JSON format:
{
  "insight": "2–3 sentence honest assessment of current standing and trajectory",
  "daily_plan": [
    "Specific actionable task 1 (30–45 min)",
    "Specific actionable task 2",
    "Specific actionable task 3",
    "Specific actionable task 4"
  ],
  "priorities": [
    "High ROI topic/action 1",
    "High ROI topic/action 2",
    "High ROI topic/action 3"
  ],
  "avoid": "What the student should deprioritize or avoid wasting time on",
  "bottleneck": "The single biggest bottleneck section/topic and why"
}

Be specific, direct, and motivating. Reference actual topics from the data."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Student data:\n{json.dumps(context, indent=2)}",
                },
            ],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        return {**_default_suggestions(), "error": str(e)}


def _default_suggestions() -> dict:
    return {
        "insight": "Log your study sessions to get personalized AI insights.",
        "daily_plan": [
            "Complete 20 QA questions across 2 topics",
            "Attempt 1 full DILR set",
            "Read 1 RC passage with full analysis",
            "Review yesterday's mistakes",
        ],
        "priorities": [
            "Start with your weakest section first each day",
            "Cover untouched topics before the exam",
            "Take at least one mock test per week",
        ],
        "avoid": "Don't spend too long on topics you're already strong in.",
        "bottleneck": "Log more data for a personalized bottleneck analysis.",
    }
