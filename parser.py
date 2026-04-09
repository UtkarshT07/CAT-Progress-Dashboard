"""
Natural language log parser using OpenAI.
Converts free-text study logs into structured JSON entries.
"""

import json
import re
from datetime import datetime
from openai import OpenAI
from cat_structure import CAT_STRUCTURE, SENTIMENT_OPTIONS, ACTIVITY_TYPES


def build_system_prompt() -> str:
    structure_str = json.dumps(CAT_STRUCTURE, indent=2)
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""You are an expert CAT exam preparation assistant.
Today's date is {today}.

Your job: parse natural language study logs into structured JSON entries.

## CAT Structure (use ONLY these values):
{structure_str}

## Rules:
1. Extract EVERY distinct study activity from the input as a separate entry.
2. Map each activity to the correct section (QA / DILR / VARC).
3. Map to the closest topic and subtopic from the structure above.
4. For DILR, use sets_attempted / correct_sets (not questions).
5. For QA and VARC, use questions_attempted / correct_answers.
6. Infer sentiment from words like "struggled", "easy", "difficult", "okay".
7. If a field is unclear, use sensible defaults (0 for counts, "medium" for sentiment).
8. Return ONLY valid JSON — no explanation, no markdown.

## Output format (array of objects):
[
  {{
    "date": "YYYY-MM-DD",
    "section": "QA | DILR | VARC",
    "topic": "<from structure>",
    "subtopic": "<from structure or closest match>",
    "activity_type": "lecture | practice | test | revision",
    "questions_attempted": 0,
    "sets_attempted": 0,
    "correct_answers": 0,
    "correct_sets": 0,
    "time_taken_minutes": 0,
    "sentiment": "easy | medium | hard | struggled"
  }}
]

Sentiments: {SENTIMENT_OPTIONS}
Activity types: {ACTIVITY_TYPES}
"""


def parse_log(api_key: str, raw_text: str) -> tuple[list[dict], str]:
    """
    Parse raw study log text into structured entries.

    Returns:
        (entries, error_message) — entries is [] on failure
    """
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": f"Parse this study log:\n{raw_text}"},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        entries = json.loads(content)

        if not isinstance(entries, list):
            entries = [entries]

        # Validate and sanitize each entry
        validated = []
        for entry in entries:
            validated.append(sanitize_entry(entry))

        return validated, ""

    except json.JSONDecodeError as e:
        return [], f"JSON parse error: {e}"
    except Exception as e:
        return [], f"API error: {e}"


def sanitize_entry(entry: dict) -> dict:
    """Ensure all required fields exist with correct types."""
    today = datetime.now().strftime("%Y-%m-%d")

    return {
        "date": entry.get("date", today) or today,
        "section": entry.get("section", "QA"),
        "topic": entry.get("topic", ""),
        "subtopic": entry.get("subtopic", ""),
        "activity_type": entry.get("activity_type", "practice"),
        "questions_attempted": max(0, int(entry.get("questions_attempted", 0) or 0)),
        "sets_attempted": max(0, int(entry.get("sets_attempted", 0) or 0)),
        "correct_answers": max(0, int(entry.get("correct_answers", 0) or 0)),
        "correct_sets": max(0, int(entry.get("correct_sets", 0) or 0)),
        "time_taken_minutes": max(0.0, float(entry.get("time_taken_minutes", 0) or 0)),
        "sentiment": entry.get("sentiment", "medium"),
    }
