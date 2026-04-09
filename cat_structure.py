"""
CAT topic taxonomy — single source of truth for all modules.
"""

CAT_STRUCTURE = {
    "QA": {
        "Arithmetic": [
            "Percentages", "Profit & Loss", "Time & Work",
            "Time Speed Distance", "Ratio & Proportion", "SI & CI"
        ],
        "Algebra": [
            "Linear Equations", "Quadratic Equations",
            "Inequalities", "Functions", "Progressions"
        ],
        "Number Systems": [
            "Divisibility", "Remainders", "HCF & LCM",
            "Factorials", "Base Systems"
        ],
        "Geometry": [
            "Lines & Angles", "Triangles", "Circles",
            "Polygons", "Mensuration", "Coordinate Geometry"
        ],
        "Modern Math": [
            "Permutation & Combination", "Probability",
            "Set Theory", "Logarithms", "Surds & Indices"
        ],
    },
    "DILR": {
        "Data Interpretation": [
            "Tables", "Bar Charts", "Line Charts",
            "Pie Charts", "Caselets", "Mixed DI"
        ],
        "Logical Reasoning": [
            "Arrangements", "Puzzles", "Games & Tournaments",
            "Venn Diagrams", "Binary Logic", "Networks"
        ],
    },
    "VARC": {
        "Reading Comprehension": [
            "Inference", "Main Idea", "Tone & Attitude",
            "Vocabulary in Context", "Paragraph Questions"
        ],
        "Para Jumbles": ["Para Jumbles"],
        "Para Summary": ["Para Summary"],
        "Odd One Out": ["Odd One Out"],
    },
}

# Flat list helpers
ALL_SECTIONS = list(CAT_STRUCTURE.keys())

def get_all_topics():
    topics = []
    for section, topic_map in CAT_STRUCTURE.items():
        for topic in topic_map:
            topics.append((section, topic))
    return topics

def get_all_subtopics():
    subtopics = []
    for section, topic_map in CAT_STRUCTURE.items():
        for topic, subs in topic_map.items():
            for sub in subs:
                subtopics.append((section, topic, sub))
    return subtopics

def get_topics_for_section(section: str):
    return list(CAT_STRUCTURE.get(section, {}).keys())

def get_subtopics_for_topic(section: str, topic: str):
    return CAT_STRUCTURE.get(section, {}).get(topic, [])

SENTIMENT_OPTIONS = ["easy", "medium", "hard", "struggled"]
ACTIVITY_TYPES = ["lecture", "practice", "test", "revision"]

# Weights for difficulty score
SENTIMENT_PENALTY = {
    "easy": 0,
    "medium": 5,
    "hard": 10,
    "struggled": 15,
}

# Speed benchmarks (seconds per unit)
SPEED_BENCHMARKS = {
    "QA":   {"good": 90, "ok": 120},   # seconds per question
    "DILR": {"good": 480, "ok": 600},  # seconds per set (~8–10 min)
    "VARC": {"good": 90, "ok": 120},   # seconds per question
}
