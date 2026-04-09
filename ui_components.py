"""
Reusable Streamlit UI components for the dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from cat_structure import CAT_STRUCTURE


# ─── Color palette ───────────────────────────────────────────────────────────

CLASSIFICATION_COLORS = {
    "Strong":    "#22c55e",   # green
    "Medium":    "#f59e0b",   # amber
    "Weak":      "#ef4444",   # red
    "Untouched": "#6b7280",   # grey
}

SECTION_COLORS = {
    "QA":   "#6366f1",   # indigo
    "DILR": "#f59e0b",   # amber
    "VARC": "#10b981",   # emerald
}


# ─── Overview metrics ────────────────────────────────────────────────────────

def render_overview_metrics(overview: dict):
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📝 Questions", f"{overview['total_questions']:,}")
    with col2:
        st.metric("🧩 Sets", f"{overview['total_sets']:,}")
    with col3:
        st.metric("🎯 Overall Accuracy",
                  f"{overview['overall_accuracy']*100:.1f}%")
    with col4:
        st.metric("⏱ Study Time",
                  f"{overview['total_time_hours']:.1f} hrs")
    with col5:
        st.metric("📅 Sessions", f"{overview['total_sessions']:,}")


# ─── Section panel ───────────────────────────────────────────────────────────

def render_section_panel(section: str, stats: dict | None):
    color = SECTION_COLORS.get(section, "#888")

    if stats is None:
        st.markdown(
            f"""<div style='border:1px solid {color}; border-radius:10px;
            padding:16px; opacity:0.5; text-align:center;'>
            <h4 style='color:{color};'>{section}</h4>
            <p>No data yet</p></div>""",
            unsafe_allow_html=True,
        )
        return

    acc = stats["accuracy"] * 100
    unit = "sets" if section == "DILR" else "questions"

    st.markdown(
        f"""<div style='border:2px solid {color}; border-radius:10px;
        padding:16px; margin-bottom:8px;'>
        <h4 style='color:{color}; margin-bottom:12px;'>{section}</h4>
        <div style='display:flex; justify-content:space-between;'>
          <span>🎯 Accuracy</span><strong>{acc:.1f}%</strong>
        </div>
        <div style='display:flex; justify-content:space-between;'>
          <span>📝 {unit.capitalize()}</span>
          <strong>{stats['attempted']} ({stats['correct']} correct)</strong>
        </div>
        <div style='display:flex; justify-content:space-between;'>
          <span>⏱ Study Time</span><strong>{stats['time_hours']:.1f} hrs</strong>
        </div>
        <div style='display:flex; justify-content:space-between;'>
          <span>📅 Sessions</span><strong>{stats['sessions']}</strong>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Topic progress bars ──────────────────────────────────────────────────────

def render_topic_progress_bars(topic_stats: pd.DataFrame, untouched: list):
    for section, topic_map in CAT_STRUCTURE.items():
        with st.expander(f"📂 {section}", expanded=True):
            for topic in topic_map:
                row = (
                    topic_stats[
                        (topic_stats["section"] == section) &
                        (topic_stats["topic"] == topic)
                    ]
                    if not topic_stats.empty else pd.DataFrame()
                )

                if row.empty:
                    # Untouched
                    st.markdown(
                        f"<div style='display:flex; align-items:center; "
                        f"margin:4px 0;'>"
                        f"<span style='width:180px; font-size:13px;'>{topic}</span>"
                        f"<div style='flex:1; background:#374151; height:16px; "
                        f"border-radius:8px;'></div>"
                        f"<span style='margin-left:8px; color:#6b7280; font-size:12px;'>"
                        f"Untouched</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    r = row.iloc[0]
                    acc = r["accuracy"] * 100
                    clf = r["classification"]
                    color = CLASSIFICATION_COLORS[clf]
                    width = max(2, int(acc))

                    st.markdown(
                        f"<div style='display:flex; align-items:center; "
                        f"margin:4px 0;'>"
                        f"<span style='width:180px; font-size:13px;'>{topic}</span>"
                        f"<div style='flex:1; background:#1f2937; height:16px; "
                        f"border-radius:8px; overflow:hidden;'>"
                        f"<div style='width:{width}%; background:{color}; "
                        f"height:100%; border-radius:8px;'></div></div>"
                        f"<span style='margin-left:8px; color:{color}; "
                        f"font-size:12px; width:60px;'>{acc:.0f}%</span>"
                        f"<span style='color:#9ca3af; font-size:11px;'>{clf}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ─── Charts ──────────────────────────────────────────────────────────────────

def render_accuracy_trend(daily: pd.DataFrame):
    if daily.empty:
        st.info("No trend data yet.")
        return

    fig = px.line(
        daily, x="date", y="avg_accuracy",
        title="📈 Daily Accuracy Trend",
        labels={"avg_accuracy": "Accuracy", "date": "Date"},
        markers=True,
        line_shape="spline",
    )
    fig.update_traces(
        line_color="#6366f1",
        marker=dict(size=8, color="#818cf8"),
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        title_font_size=14,
    )
    fig.add_hline(
        y=0.80, line_dash="dash",
        line_color="#22c55e", opacity=0.5,
        annotation_text="Target 80%",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_speed_trend(daily: pd.DataFrame):
    if daily.empty:
        st.info("No trend data yet.")
        return

    fig = px.bar(
        daily, x="date", y="total_time",
        title="⏱ Daily Study Time (minutes)",
        labels={"total_time": "Minutes", "date": "Date"},
        color_discrete_sequence=["#f59e0b"],
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        title_font_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_attempts_trend(daily: pd.DataFrame):
    if daily.empty:
        st.info("No trend data yet.")
        return

    fig = px.area(
        daily, x="date", y="total_units",
        title="📝 Daily Attempts Trend",
        labels={"total_units": "Questions / Sets", "date": "Date"},
        color_discrete_sequence=["#10b981"],
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        title_font_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_weakest_topics_chart(topic_stats: pd.DataFrame):
    if topic_stats.empty:
        st.info("Log some sessions to see weak areas.")
        return

    weak = topic_stats.nlargest(8, "avg_difficulty")
    if weak.empty:
        st.info("No weak topics identified yet.")
        return

    weak["label"] = weak["section"] + " → " + weak["topic"]
    weak["color"] = weak["classification"].map(CLASSIFICATION_COLORS)

    fig = px.bar(
        weak,
        x="avg_difficulty", y="label",
        orientation="h",
        title="🔴 Weakest Areas (by Difficulty Score)",
        labels={"avg_difficulty": "Difficulty Score", "label": "Topic"},
        color="classification",
        color_discrete_map=CLASSIFICATION_COLORS,
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        yaxis={"categoryorder": "total ascending"},
        title_font_size=14,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Readiness visualization ─────────────────────────────────────────────────

def render_readiness_gauge(score: float, level: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 32, "color": "#e2e8f0"}},
        title={"text": level, "font": {"size": 18, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
            "bar": {"color": _score_color(score)},
            "bgcolor": "#1f2937",
            "bordercolor": "#374151",
            "steps": [
                {"range": [0, 40],  "color": "#1f2937"},
                {"range": [40, 70], "color": "#1f2937"},
                {"range": [70, 85], "color": "#1f2937"},
                {"range": [85, 100],"color": "#1f2937"},
            ],
            "threshold": {
                "line": {"color": "#22c55e", "width": 4},
                "thickness": 0.85,
                "value": 85,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        height=280,
        margin=dict(t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_readiness_history(history: pd.DataFrame, projection: dict):
    if history.empty and not projection.get("projection_dates"):
        st.info("Build readiness history by logging daily sessions.")
        return

    fig = go.Figure()

    # Historical line
    if not history.empty:
        fig.add_trace(go.Scatter(
            x=history["date"],
            y=history["readiness_score"],
            mode="lines+markers",
            name="Actual Score",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=7),
        ))

    # Projection line
    if projection.get("projection_dates"):
        fig.add_trace(go.Scatter(
            x=projection["projection_dates"],
            y=projection["projection_scores"],
            mode="lines",
            name="Projected",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ))

    # Target line
    fig.add_hline(
        y=85, line_dash="dot",
        line_color="#22c55e",
        annotation_text="Exam Ready (85)",
        annotation_position="right",
    )

    # Milestone bands
    for threshold, label, color in [
        (40, "Developing", "#1e3a5f"),
        (70, "Competitive", "#1a3a2a"),
        (85, "Exam Ready", "#1a3a1a"),
    ]:
        fig.add_hrect(
            y0=threshold - 30 if threshold > 40 else 0,
            y1=threshold,
            fillcolor=color,
            opacity=0.15,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
        )

    fig.update_layout(
        title="📊 Readiness Journey & Projection",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        yaxis_range=[0, 105],
        title_font_size=14,
        legend=dict(bgcolor="#1f2937"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_readiness_progress_bar(score: float):
    pct = int(score)
    color = _score_color(score)
    st.markdown(f"""
    <div style='margin:8px 0;'>
      <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
        <span style='font-size:13px; color:#9ca3af;'>Journey to Exam Readiness</span>
        <span style='font-size:13px; color:{color};'>{score:.1f} / 85+ target</span>
      </div>
      <div style='background:#1f2937; border-radius:10px; overflow:hidden; height:20px;'>
        <div style='width:{pct}%; background:linear-gradient(90deg, #6366f1, {color});
             height:100%; border-radius:10px; transition:width 0.5s;'></div>
      </div>
      <div style='display:flex; justify-content:space-between; margin-top:4px;
           font-size:11px; color:#6b7280;'>
        <span>0 Early</span><span>40 Developing</span>
        <span>70 Competitive</span><span>85 Ready</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _score_color(score: float) -> str:
    if score >= 85:
        return "#22c55e"
    elif score >= 70:
        return "#10b981"
    elif score >= 40:
        return "#f59e0b"
    else:
        return "#ef4444"


# ─── Suggestion cards ────────────────────────────────────────────────────────

def render_suggestions(suggestions: dict):
    st.markdown(f"""
    <div style='background:#1e1b4b; border:1px solid #4338ca;
    border-radius:12px; padding:20px; margin-bottom:16px;'>
      <h4 style='color:#818cf8; margin-bottom:8px;'>🤖 AI Insight</h4>
      <p style='color:#e2e8f0; line-height:1.6;'>{suggestions.get('insight', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Today's Recommended Plan")
        for i, task in enumerate(suggestions.get("daily_plan", []), 1):
            st.markdown(f"**{i}.** {task}")

        st.markdown("#### 🚨 Bottleneck")
        st.warning(suggestions.get("bottleneck", "—"))

    with col2:
        st.markdown("#### 🎯 High Priority Topics")
        for p in suggestions.get("priorities", []):
            st.markdown(f"• {p}")

        st.markdown("#### ⛔ Avoid / Deprioritize")
        st.info(suggestions.get("avoid", "—"))


# ─── Log table ───────────────────────────────────────────────────────────────

def render_log_table(df: pd.DataFrame):
    if df.empty:
        st.info("No logs found for the selected filters.")
        return

    display_cols = [
        "date", "section", "topic", "subtopic", "activity_type",
        "questions_attempted", "sets_attempted",
        "correct_answers", "correct_sets",
        "time_taken_minutes", "sentiment",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].rename(columns={
            "time_taken_minutes": "time (min)",
            "questions_attempted": "q_attempted",
            "correct_answers": "q_correct",
            "sets_attempted": "sets_attempted",
            "correct_sets": "sets_correct",
        }),
        use_container_width=True,
        hide_index=True,
    )
