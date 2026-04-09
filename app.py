"""
CAT Preparation Dashboard — Main Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

# ── Local modules ─────────────────────────────────────────────────────────────
from database import (
    initialize_db, insert_logs, fetch_all_logs,
    save_readiness_snapshot, fetch_readiness_history, delete_log,
)
from parser import parse_log
from metrics import (
    compute_overview, compute_topic_stats, compute_daily_trend,
    get_weakest_topics, get_untouched_topics, get_section_stats,
)
from readiness import compute_readiness_scores, estimate_time_to_readiness
from suggestions import generate_suggestions
from ui_components import (
    render_overview_metrics, render_section_panel,
    render_topic_progress_bars, render_accuracy_trend,
    render_speed_trend, render_attempts_trend,
    render_weakest_topics_chart, render_readiness_gauge,
    render_readiness_history, render_readiness_progress_bar,
    render_suggestions, render_log_table,
    SECTION_COLORS,
)
from cat_structure import ALL_SECTIONS, get_topics_for_section


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAT Prep Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme override ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f172a; color: #e2e8f0; }
  .stTextArea textarea { background-color: #1e293b !important; color: #e2e8f0 !important; }
  .stSelectbox > div { background-color: #1e293b; }
  section[data-testid="stSidebar"] { background-color: #1e293b; }
  .stMetric { background-color: #1e293b; border-radius: 10px; padding: 12px; }
  div[data-testid="stMetricValue"] { color: #818cf8; font-size: 1.6rem; }
  .stExpander { background-color: #1e293b; border: 1px solid #374151; border-radius: 8px; }
  h1, h2, h3, h4 { color: #e2e8f0; }
  .stTabs [data-baseweb="tab"] { background-color: #1e293b; }
  .stAlert { border-radius: 8px; }
  hr { border-color: #374151; }
</style>
""", unsafe_allow_html=True)


# ── Initialize ────────────────────────────────────────────────────────────────
initialize_db()


# ── Session state ─────────────────────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "suggestions_cache" not in st.session_state:
    st.session_state["suggestions_cache"] = None
if "last_parsed" not in st.session_state:
    st.session_state["last_parsed"] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🎯 CAT Prep")
    st.markdown("---")

    # API Key
    st.markdown("### 🔑 OpenAI API Key")
    api_key_input = st.text_input(
        "Enter your key",
        type="password",
        value=st.session_state["api_key"],
        help="Your key stays in memory only — never stored.",
    )
    if api_key_input:
        st.session_state["api_key"] = api_key_input

    st.markdown("---")

    # ── Log Input ────────────────────────────────────────────────────────────
    st.markdown("### 📝 Log Your Session")
    log_input = st.text_area(
        "Describe what you studied",
        placeholder=(
            "e.g. Did 20 questions of Time & Work, got 12 correct, "
            "took 40 mins. Struggled. Also solved 2 DI sets with 1 correct."
        ),
        height=140,
    )

    if st.button("⚡ Parse & Save", use_container_width=True, type="primary"):
        if not st.session_state["api_key"]:
            st.error("Please enter your OpenAI API key first.")
        elif not log_input.strip():
            st.warning("Please enter a study log.")
        else:
            with st.spinner("Parsing your log..."):
                entries, error = parse_log(
                    st.session_state["api_key"], log_input
                )

            if error:
                st.error(f"Parse error: {error}")
            elif not entries:
                st.warning("No entries were extracted. Try rephrasing.")
            else:
                count = insert_logs(entries, raw_input=log_input)
                st.session_state["last_parsed"] = entries
                st.session_state["suggestions_cache"] = None  # invalidate cache
                st.success(f"✅ Saved {count} session(s)!")

                # Show parsed preview
                with st.expander("🔍 Parsed Entries", expanded=True):
                    for e in entries:
                        sec_color = SECTION_COLORS.get(e.get("section",""), "#888")
                        st.markdown(
                            f"<span style='color:{sec_color}; font-weight:bold;'>"
                            f"{e.get('section')}</span> → "
                            f"**{e.get('topic')}** / {e.get('subtopic','—')}<br>"
                            f"Accuracy: {e.get('correct_answers',0) or e.get('correct_sets',0)}"
                            f"/{e.get('questions_attempted',0) or e.get('sets_attempted',0)} | "
                            f"Time: {e.get('time_taken_minutes',0)} min | "
                            f"Mood: {e.get('sentiment','—')}",
                            unsafe_allow_html=True,
                        )
                        st.markdown("---")
                st.rerun()

    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Filters")

    date_range = st.date_input(
        "Date range",
        value=[date.today() - timedelta(days=30), date.today()],
    )
    start_date = date_range[0].isoformat() if len(date_range) > 0 else None
    end_date = date_range[1].isoformat() if len(date_range) > 1 else None

    section_filter = st.selectbox("Section", ["All"] + ALL_SECTIONS)

    topics_for_section = (
        get_topics_for_section(section_filter)
        if section_filter != "All" else []
    )
    topic_filter = st.selectbox(
        "Topic",
        ["All"] + topics_for_section,
        disabled=(section_filter == "All"),
    )

    st.markdown("---")
    st.markdown(
        "<center style='color:#6b7280; font-size:11px;'>"
        "CAT Dashboard • Powered by OpenAI</center>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH DATA
# ═══════════════════════════════════════════════════════════════════════════════

df_filtered = fetch_all_logs(
    start_date=start_date,
    end_date=end_date,
    section=section_filter,
    topic=topic_filter if topic_filter != "All" else None,
)

# Always fetch all data for readiness (needs full history)
df_all = fetch_all_logs()

overview = compute_overview(df_filtered)
topic_stats = compute_topic_stats(df_filtered)
section_stats = get_section_stats(df_filtered)
daily_trend = compute_daily_trend(df_filtered)
untouched = get_untouched_topics(df_filtered)

# Readiness uses full data
readiness = compute_readiness_scores(df_all)
readiness_history = fetch_readiness_history()
time_est = estimate_time_to_readiness(readiness_history, readiness["readiness_score"])

# Save today's snapshot
from datetime import datetime
save_readiness_snapshot(datetime.now().strftime("%Y-%m-%d"), readiness)
readiness_history = fetch_readiness_history()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🎯 CAT Preparation Dashboard")
st.markdown(
    f"<span style='color:#6b7280;'>Filters: "
    f"{section_filter} | {start_date} → {end_date}</span>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Overview",
    "📚 Topics",
    "📈 Trends",
    "🏆 Readiness",
    "🤖 AI Coach",
    "📋 Logs",
])


# ─── Tab 1: Overview ──────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## 📊 Overview")
    render_overview_metrics(overview)

    st.markdown("---")
    st.markdown("## 🗂 Section Panels")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_section_panel("QA", section_stats.get("QA"))
    with col2:
        render_section_panel("DILR", section_stats.get("DILR"))
    with col3:
        render_section_panel("VARC", section_stats.get("VARC"))

    st.markdown("---")

    # Weakest / Needs Attention / Strong / Untouched
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🔴 Weakest Areas")
        if not topic_stats.empty:
            weak = topic_stats[topic_stats["classification"] == "Weak"]
            if weak.empty:
                st.success("No weak areas detected yet!")
            else:
                for _, r in weak.nlargest(5, "avg_difficulty").iterrows():
                    st.markdown(
                        f"🔴 **{r['section']} → {r['topic']}** — "
                        f"{r['accuracy']*100:.0f}% accuracy | "
                        f"Difficulty: {r['avg_difficulty']:.0f}"
                    )
        else:
            st.info("Log sessions to identify weak areas.")

        st.markdown("### 🟡 Needs Attention")
        if not topic_stats.empty:
            medium = topic_stats[topic_stats["classification"] == "Medium"]
            if medium.empty:
                st.info("No medium topics.")
            else:
                for _, r in medium.iterrows():
                    st.markdown(
                        f"🟡 **{r['section']} → {r['topic']}** — "
                        f"{r['accuracy']*100:.0f}% accuracy"
                    )

    with col_right:
        st.markdown("### 🟢 Strong Areas")
        if not topic_stats.empty:
            strong = topic_stats[topic_stats["classification"] == "Strong"]
            if strong.empty:
                st.info("Keep going — strong topics will appear here.")
            else:
                for _, r in strong.iterrows():
                    st.markdown(
                        f"🟢 **{r['section']} → {r['topic']}** — "
                        f"{r['accuracy']*100:.0f}% accuracy"
                    )

        st.markdown("### ⬜ Untouched Topics")
        if untouched:
            for t in untouched[:8]:
                st.markdown(f"⬜ **{t['section']} → {t['topic']}**")
            if len(untouched) > 8:
                st.caption(f"... and {len(untouched)-8} more")
        else:
            st.success("All topics touched! Great coverage.")


# ─── Tab 2: Topics ────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📚 Topic Progress")

    col_chart, col_bars = st.columns([2, 1])
    with col_chart:
        render_weakest_topics_chart(topic_stats)
    with col_bars:
        if not topic_stats.empty:
            clf_counts = topic_stats["classification"].value_counts()
            import plotly.express as px
            fig = px.pie(
                values=clf_counts.values,
                names=clf_counts.index,
                title="Topic Distribution",
                color=clf_counts.index,
                color_discrete_map={
                    "Strong": "#22c55e",
                    "Medium": "#f59e0b",
                    "Weak": "#ef4444",
                    "Untouched": "#6b7280",
                },
            )
            fig.update_layout(
                paper_bgcolor="#0f172a",
                font_color="#e2e8f0",
                title_font_size=14,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## 📊 Topic Progress Bars")
    render_topic_progress_bars(topic_stats, untouched)


# ─── Tab 3: Trends ────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 📈 Performance Trends")
    render_accuracy_trend(daily_trend)

    col1, col2 = st.columns(2)
    with col1:
        render_speed_trend(daily_trend)
    with col2:
        render_attempts_trend(daily_trend)

    # Heatmap of study activity (calendar-style)
    if not df_all.empty:
        st.markdown("---")
        st.markdown("### 📅 Study Activity Heatmap")
        daily_counts = (
            df_all.groupby("date")["id"].count()
            .reset_index()
            .rename(columns={"id": "sessions"})
        )
        daily_counts["date"] = pd.to_datetime(daily_counts["date"])

        import plotly.express as px
        fig = px.scatter(
            daily_counts, x="date", y="sessions",
            size="sessions", color="sessions",
            color_continuous_scale="Viridis",
            title="Sessions Per Day",
        )
        fig.update_layout(
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Tab 4: Readiness ─────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🏆 Exam Readiness")

    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        render_readiness_gauge(
            readiness["readiness_score"], readiness["level"]
        )
        render_readiness_progress_bar(readiness["readiness_score"])

    with col_details:
        st.markdown("### Score Breakdown")
        components = [
            ("🎯 Accuracy", readiness["accuracy_score"], 40),
            ("⚡ Speed", readiness["speed_score"], 20),
            ("📚 Coverage", readiness["coverage_score"], 25),
            ("📅 Consistency", readiness["consistency_score"], 15),
        ]
        for label, score, max_score in components:
            pct = (score / max_score * 100) if max_score > 0 else 0
            st.markdown(
                f"<div style='margin:8px 0;'>"
                f"<div style='display:flex; justify-content:space-between;'>"
                f"<span>{label}</span>"
                f"<span style='color:#818cf8;'>{score:.1f}/{max_score}</span></div>"
                f"<div style='background:#1f2937; border-radius:6px; height:10px; overflow:hidden;'>"
                f"<div style='width:{pct:.0f}%; background:#6366f1; height:100%; "
                f"border-radius:6px;'></div></div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Time estimate
        w_min = time_est.get("weeks_min")
        w_max = time_est.get("weeks_max")
        rate = time_est.get("weekly_rate")

        if w_min and w_max:
            st.markdown("### ⏳ Time to Exam Readiness")
            st.markdown(
                f"<div style='background:#1e3a5f; border-radius:10px; "
                f"padding:16px; text-align:center;'>"
                f"<h2 style='color:#60a5fa; margin:0;'>{w_min}–{w_max} weeks</h2>"
                f"<p style='color:#93c5fd; margin:4px 0;'>to reach Exam Ready (85+)</p>"
                f"{'<p style=color:#6b7280;font-size:12px;>Improving at ' + str(round(rate,1)) + ' pts/week</p>' if rate else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Log at least 2 sessions to get a time estimate.")

    st.markdown("---")
    render_readiness_history(readiness_history, time_est)


# ─── Tab 5: AI Coach ──────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 🤖 AI Study Coach")

    if not st.session_state["api_key"]:
        st.warning("Enter your OpenAI API key in the sidebar to activate the AI Coach.")
    else:
        if st.button("🔄 Generate Fresh Suggestions", type="primary"):
            with st.spinner("Consulting AI coach..."):
                suggestions = generate_suggestions(
                    st.session_state["api_key"],
                    df_all,
                    readiness,
                    time_est,
                )
                st.session_state["suggestions_cache"] = suggestions

        if st.session_state["suggestions_cache"]:
            if "error" in st.session_state["suggestions_cache"]:
                st.error(
                    f"AI error: {st.session_state['suggestions_cache']['error']}"
                )
            render_suggestions(st.session_state["suggestions_cache"])
        else:
            st.info(
                "Click **Generate Fresh Suggestions** to get personalized "
                "coaching based on your logs."
            )

            # Show default plan if no cache
            st.markdown("### 📋 Default Daily Plan")
            st.markdown("""
            1. **QA Practice** — 20 questions from your weakest topic (40 min)
            2. **DILR Set** — Attempt 1 full set with detailed analysis (20 min)
            3. **VARC RC** — Read 1 passage, answer all questions (20 min)
            4. **Error Log Review** — Revisit yesterday's mistakes (15 min)
            """)


# ─── Tab 6: Logs ──────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 📋 Session Logs")

    col_search, col_delete = st.columns([3, 1])
    with col_search:
        st.caption(
            f"Showing {len(df_filtered)} log entries "
            f"(filtered) | {len(df_all)} total"
        )

    with col_delete:
        delete_id = st.number_input(
            "Delete log ID", min_value=1, step=1, value=1
        )
        if st.button("🗑 Delete", type="secondary"):
            delete_log(int(delete_id))
            st.success(f"Deleted log #{delete_id}")
            st.rerun()

    render_log_table(df_filtered)

    # Export
    if not df_filtered.empty:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            "📥 Export as CSV",
            data=csv,
            file_name="cat_logs.csv",
            mime="text/csv",
        )
