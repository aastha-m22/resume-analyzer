"""
app.py — Resume Analyzer: Streamlit web application entry point.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils import (
    extract_text_from_pdf,
    parse_resume,
    compute_match_score,
    get_missing_keywords,
)
from analyzer import generate_resume_feedback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Resume Analyzer AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, modern, professional dark-accent design
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Remove default Streamlit top padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid #334155;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.4);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }

    /* Section cards */
    .section-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748b;
        margin-bottom: 0.8rem;
    }

    /* Score display */
    .score-ring-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .score-number {
        font-size: 4rem;
        font-weight: 700;
        font-family: 'DM Mono', monospace;
        line-height: 1;
    }
    .score-label { color: #94a3b8; font-size: 0.9rem; margin-top: 4px; }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 3px;
        font-family: 'DM Mono', monospace;
    }
    .missing-tag {
        background: rgba(239,68,68,0.1);
        border-color: rgba(239,68,68,0.3);
        color: #fca5a5;
    }

    /* Feedback items */
    .feedback-item {
        background: #0f172a;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .feedback-item.strength { border-left-color: #22c55e; }
    .feedback-item.suggestion { border-left-color: #f59e0b; }
    .feedback-item.missing { border-left-color: #ef4444; }

    /* Progress bar override */
    .stProgress > div > div > div { background: #6366f1 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 20px rgba(99,102,241,0.35) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Text area */
    .stTextArea textarea {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Info box */
    .info-box {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #c7d2fe;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .info-box .info-icon { font-size: 1.2rem; margin-right: 6px; }

    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.1));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.7;
        font-style: italic;
    }

    /* Divider */
    hr { border-color: #1e293b !important; }

    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'DM Mono', monospace;
    }
    .metric-label { color: #64748b; font-size: 0.8rem; margin-top: 2px; }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        background: #1e293b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper rendering functions
# ---------------------------------------------------------------------------

def render_score(score: float):
    """Render a visual match score with color coding."""
    if score >= 75:
        color = "#22c55e"
        verdict = "Excellent Match 🎯"
    elif score >= 50:
        color = "#f59e0b"
        verdict = "Good Match ✨"
    elif score >= 30:
        color = "#f97316"
        verdict = "Moderate Match 📈"
    else:
        color = "#ef4444"
        verdict = "Needs Improvement 🔧"

    st.markdown(
        f"""
        <div class="score-ring-container">
            <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;
                        color:#64748b;margin-bottom:0.5rem;">Match Score</div>
            <div class="score-number" style="color:{color}">{score}%</div>
            <div style="color:{color};font-weight:600;margin-top:0.4rem;font-size:0.95rem">{verdict}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(score), 100))


def render_tags(items: list, tag_class="skill-tag"):
    """Render a list of items as inline tags."""
    if not items:
        st.markdown("<em style='color:#475569'>None detected</em>", unsafe_allow_html=True)
        return
    html = "".join(f'<span class="{tag_class}">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def render_feedback_list(items: list, css_class: str):
    """Render a list of feedback strings as styled cards."""
    if not items:
        st.markdown("<em style='color:#475569'>No items found.</em>", unsafe_allow_html=True)
        return
    for item in items:
        st.markdown(
            f'<div class="feedback-item {css_class}">• {item}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------

def main():
    # ── Hero Header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-badge">⚡ Powered by NLP + AI</div>
            <div class="hero-title">Resume Analyzer</div>
            <div class="hero-subtitle">
                Upload your resume, paste a job description, and get AI-powered
                insights on your fit — in seconds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Check API key ─────────────────────────────────────────────────────────
    has_key = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not has_key:
        st.markdown(
            """
            <div class="info-box">
                <span class="info-icon">⚠️</span>
                <strong>No API key detected.</strong> Set <code>GROQ_API_KEY</code> or
                <code>OPENAI_API_KEY</code> in your <code>.env</code> file to enable AI feedback.
                NLP scoring will still work without a key.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

    # ── Two-column input layout ───────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Drag & drop your PDF resume here",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.success(f"✅ Loaded: **{uploaded_file.name}**")

    with col_right:
        st.markdown("### 💼 Job Description")
        job_description = st.text_area(
            "Paste the full job description here",
            height=180,
            placeholder="e.g. We are looking for a Senior Python Developer with 5+ years experience in "
                        "Django, REST APIs, PostgreSQL, AWS, and CI/CD pipelines...",
            label_visibility="collapsed",
        )

    # ── Analyze Button ────────────────────────────────────────────────────────
    st.write("")
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyze_clicked = st.button("🔍 Analyze Resume", use_container_width=True)

    st.write("")

    # ── Analysis Logic ────────────────────────────────────────────────────────
    if analyze_clicked:
        if not uploaded_file:
            st.error("❌ Please upload a PDF resume first.")
            return
        if not job_description.strip():
            st.error("❌ Please paste a job description.")
            return

        with st.spinner("Parsing resume and running NLP analysis..."):
            # 1. Extract text from PDF
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
            except Exception as e:
                st.error(f"Failed to parse PDF: {e}")
                return

            if len(resume_text.strip()) < 50:
                st.error("Could not extract readable text from the PDF. Is it a scanned image?")
                return

            # 2. NLP extraction
            parsed = parse_resume(resume_text)

            # 3. Similarity score
            match_score = compute_match_score(resume_text, job_description)

            # 4. Missing keywords
            missing_kw = get_missing_keywords(resume_text, job_description)

        # ── Results Layout ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        st.write("")

        # Top row: score + quick metrics
        score_col, metrics_col = st.columns([1, 2], gap="large")

        with score_col:
            render_score(match_score)

        with metrics_col:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{len(parsed["skills"])}</div>'
                    f'<div class="metric-label">Skills Found</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{len(parsed["education"])}</div>'
                    f'<div class="metric-label">Education Items</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{len(parsed["experience"])}</div>'
                    f'<div class="metric-label">Experience Items</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown(
                f'<div class="info-box">'
                f'<strong>📬 Contact:</strong> {parsed["contact"]["email"]} &nbsp;|&nbsp; '
                f'📞 {parsed["contact"]["phone"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.write("")

        # Extracted data expander
        with st.expander("🔬 View Extracted Resume Data", expanded=True):
            exp1, exp2, exp3 = st.columns(3)

            with exp1:
                st.markdown('<div class="section-title">🛠 Skills Detected</div>', unsafe_allow_html=True)
                render_tags(parsed["skills"], "skill-tag")

            with exp2:
                st.markdown('<div class="section-title">🎓 Education</div>', unsafe_allow_html=True)
                if parsed["education"]:
                    for edu in parsed["education"]:
                        st.markdown(f"• {edu}")
                else:
                    st.markdown("<em style='color:#475569'>None detected</em>", unsafe_allow_html=True)

            with exp3:
                st.markdown('<div class="section-title">💼 Experience</div>', unsafe_allow_html=True)
                if parsed["experience"]:
                    for exp in parsed["experience"]:
                        st.markdown(f"• {exp}")
                else:
                    st.markdown("<em style='color:#475569'>None detected</em>", unsafe_allow_html=True)

        st.write("")

        # Missing keywords
        st.markdown("### 🔑 Missing Keywords from Job Description")
        render_tags(missing_kw[:18], "skill-tag missing-tag")

        st.write("")

        # AI Feedback Section
        st.markdown("### 🤖 AI-Powered Feedback")

        if not has_key:
            st.info("💡 Add a GROQ_API_KEY or OPENAI_API_KEY to your `.env` file to unlock AI suggestions.")
        else:
            with st.spinner("Generating AI feedback — this takes ~10 seconds..."):
                try:
                    feedback = generate_resume_feedback(
                        resume_text=resume_text,
                        job_description=job_description,
                        parsed_data=parsed,
                        match_score=match_score,
                        missing_keywords=missing_kw,
                    )
                except Exception as e:
                    st.error(f"AI feedback failed: {e}")
                    feedback = None

            if feedback:
                # Overall summary
                if feedback.get("overall_summary"):
                    st.markdown(
                        f'<div class="summary-box">💬 {feedback["overall_summary"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")

                fb1, fb2, fb3 = st.columns(3)

                with fb1:
                    st.markdown("#### ✅ Strengths")
                    render_feedback_list(feedback.get("strengths", []), "strength")

                with fb2:
                    st.markdown("#### ⚠️ Missing Skills")
                    render_feedback_list(feedback.get("missing_skills", []), "missing")

                with fb3:
                    st.markdown("#### 💡 Suggestions")
                    render_feedback_list(feedback.get("suggestions", []), "suggestion")

        # Raw text expander
        with st.expander("📃 View Extracted Resume Text"):
            st.code(resume_text[:3000], language=None)

        st.write("")
        st.markdown(
            "<div style='text-align:center;color:#475569;font-size:0.8rem'>"
            "Resume Analyzer AI · Built with spaCy, scikit-learn & LLM APIs"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
