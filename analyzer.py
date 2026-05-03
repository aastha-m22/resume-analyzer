"""
analyzer.py — LLM integration for AI-powered resume feedback.
Supports both OpenAI and Groq APIs via a unified interface.
"""

import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# Client factory — auto-selects OpenAI or Groq based on env vars
# ---------------------------------------------------------------------------

def _get_client() -> tuple[OpenAI, str]:
    """
    Returns (client, model_name) by checking available API keys.
    Priority: Groq → OpenAI.
    Groq uses the OpenAI-compatible SDK with a different base_url.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if groq_key:
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = "llama3-8b-8192"  # fast, free-tier Groq model
        return client, model

    if openai_key:
        client = OpenAI(api_key=openai_key)
        model = "gpt-3.5-turbo"
        return client, model

    raise EnvironmentError(
        "No API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
    )


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
    """
    Generic LLM call wrapper with error handling.
    Returns the model's text response.
    """
    client, model = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Resume Analysis — AI Feedback
# ---------------------------------------------------------------------------

def generate_resume_feedback(
    resume_text: str,
    job_description: str,
    parsed_data: dict,
    match_score: float,
    missing_keywords: list[str],
) -> dict:
    """
    Generate structured AI feedback using an LLM.
    Returns a dict with:
        - strengths (list of strings)
        - missing_skills (list of strings)
        - suggestions (list of strings)
        - overall_summary (string)
    """
    system_prompt = """You are an expert career coach and technical recruiter with 15+ years of experience.
Analyze resumes against job descriptions and provide actionable, specific feedback.
Be concise, direct, and professional. Your goal is to help the candidate improve their chances."""

    user_prompt = f"""
Analyze this candidate's resume against the job description and provide structured feedback.

--- JOB DESCRIPTION ---
{job_description[:2000]}

--- RESUME (Extracted Text) ---
{resume_text[:2500]}

--- MATCH SCORE ---
{match_score}% (TF-IDF cosine similarity)

--- DETECTED SKILLS ---
{", ".join(parsed_data.get("skills", [])) or "None detected"}

--- MISSING KEYWORDS FROM JD ---
{", ".join(missing_keywords[:15]) or "None"}

Respond ONLY with the following JSON structure (no extra text, no markdown fences):
{{
  "strengths": ["strength 1", "strength 2", "strength 3", "strength 4"],
  "missing_skills": ["skill 1", "skill 2", "skill 3", "skill 4", "skill 5"],
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4"],
  "overall_summary": "2-3 sentence professional summary of fit for this role."
}}
"""

    raw = _call_llm(system_prompt, user_prompt, max_tokens=900)

    # Parse JSON safely
    import json
    import re

    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract manually from raw text
        result = {
            "strengths": ["See raw AI response below"],
            "missing_skills": missing_keywords[:5],
            "suggestions": ["Review the full analysis"],
            "overall_summary": raw[:500],
        }

    return result


# ---------------------------------------------------------------------------
# Quick keyword gap analysis (lightweight alternative to full LLM call)
# ---------------------------------------------------------------------------

def generate_keyword_suggestions(resume_text: str, job_description: str) -> str:
    """
    Generate targeted keyword suggestions to improve ATS score.
    Returns a short plain-text response.
    """
    system_prompt = "You are an ATS optimization expert. Be brief and specific."
    user_prompt = f"""
Given this job description snippet:
{job_description[:800]}

And this resume snippet:
{resume_text[:800]}

List the TOP 5 keywords the candidate should add to pass ATS filters.
Format: numbered list, one per line, max 10 words each.
"""
    return _call_llm(system_prompt, user_prompt, max_tokens=300)
