"""
utils.py — PDF parsing, NLP extraction, and similarity scoring utilities.
"""

import re
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Load spaCy model (en_core_web_sm must be downloaded separately)
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------------
# Curated skills keyword list — expand as needed
# ---------------------------------------------------------------------------
SKILLS_DB = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "rust", "kotlin", "swift", "scala", "r", "matlab", "php", "bash", "shell",
    # Web frameworks
    "react", "angular", "vue", "django", "flask", "fastapi", "node.js", "express",
    "spring", "rails", "nextjs", "nuxtjs", "svelte",
    # Data & ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "opencv", "nltk", "spacy", "hugging face", "transformers", "langchain",
    "xgboost", "lightgbm", "matplotlib", "seaborn", "plotly",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "oracle", "firebase",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "jenkins", "github actions", "gitlab ci", "linux", "git",
    # Data Engineering
    "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",
    "databricks", "etl", "data pipeline",
    # AI & LLMs
    "openai", "gpt", "llm", "prompt engineering", "rag", "vector database",
    "pinecone", "chromadb", "weaviate", "fine-tuning",
    # Soft skills / methodologies
    "agile", "scrum", "rest api", "graphql", "microservices", "oop",
    "tdd", "unit testing", "devops", "mlops",
]

# ---------------------------------------------------------------------------
# PDF Parsing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from an uploaded PDF file object (Streamlit UploadedFile).
    Uses PyMuPDF for clean text extraction.
    """
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(text_parts).strip()


# ---------------------------------------------------------------------------
# NLP Extraction
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> list[str]:
    """
    Extract skills by matching text tokens against SKILLS_DB.
    Case-insensitive multi-word matching is also supported.
    """
    text_lower = text.lower()
    found = []
    for skill in SKILLS_DB:
        # Match whole word / phrase
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill.title())
    return sorted(set(found))


def extract_education(text: str) -> list[str]:
    """
    Extract education details using regex patterns and spaCy ORG/DATE entities.
    Looks for degree keywords and institution names.
    """
    lines = text.split("\n")
    education = []

    degree_patterns = [
        r"\b(B\.?Tech|B\.?E|B\.?Sc|B\.?A|M\.?Tech|M\.?Sc|M\.?A|MBA|Ph\.?D|Bachelor|Master|Associate|Diploma)\b",
    ]

    for line in lines:
        for pattern in degree_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                clean = line.strip()
                if clean and len(clean) > 5:
                    education.append(clean)
                break

    # Also harvest ORG entities that appear near education keywords
    doc = nlp(text[:5000])  # limit for speed
    edu_keywords = {"university", "college", "institute", "school", "academy"}
    for ent in doc.ents:
        if ent.label_ == "ORG":
            if any(kw in ent.text.lower() for kw in edu_keywords):
                education.append(ent.text.strip())

    # Deduplicate while preserving order
    seen = set()
    unique_edu = []
    for item in education:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique_edu.append(item)
    return unique_edu[:8]  # top 8 entries


def extract_experience(text: str) -> list[str]:
    """
    Extract job experience entries using regex and spaCy NER.
    Identifies lines containing job titles, company names, and date ranges.
    """
    lines = text.split("\n")
    experience = []

    # Regex: lines with year ranges suggest work experience entries
    year_range_pattern = r"\b(19|20)\d{2}\s*[-–—to]+\s*((19|20)\d{2}|present|current|now)\b"

    for i, line in enumerate(lines):
        line = line.strip()
        if re.search(year_range_pattern, line, re.IGNORECASE) and len(line) > 10:
            # Include surrounding context (previous or same line)
            entry = line
            if i > 0 and lines[i - 1].strip():
                entry = lines[i - 1].strip() + " | " + line
            experience.append(entry)

    # Fallback: spaCy ORG + DATE entity pairs
    if len(experience) < 2:
        doc = nlp(text[:5000])
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        for org in orgs[:6]:
            experience.append(org)
        for date in dates[:4]:
            experience.append(date)

    # Deduplicate
    seen = set()
    unique_exp = []
    for item in experience:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique_exp.append(item)
    return unique_exp[:10]


def extract_contact_info(text: str) -> dict:
    """Extract email and phone from resume text."""
    email_pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"(\+?\d[\d\s\-().]{7,}\d)"

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    return {
        "email": emails[0] if emails else "Not found",
        "phone": phones[0].strip() if phones else "Not found",
    }


def parse_resume(text: str) -> dict:
    """
    Master function: parse all structured fields from resume text.
    Returns a dict with skills, education, experience, and contact info.
    """
    return {
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "contact": extract_contact_info(text),
    }


# ---------------------------------------------------------------------------
# Similarity Scoring (TF-IDF + Cosine Similarity)
# ---------------------------------------------------------------------------

def compute_match_score(resume_text: str, job_description: str) -> float:
    """
    Compute cosine similarity between resume and job description
    using TF-IDF vectorization.
    Returns a score between 0 and 100.
    """
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)


def get_missing_keywords(resume_text: str, job_description: str) -> list[str]:
    """
    Find important keywords in the job description that are absent in the resume.
    Uses TF-IDF to identify the most significant JD terms.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50, ngram_range=(1, 2))
    vectorizer.fit([job_description])
    jd_terms = set(vectorizer.get_feature_names_out())

    resume_lower = resume_text.lower()
    missing = [term for term in jd_terms if term.lower() not in resume_lower]
    return sorted(missing)[:20]
