# 📄 Resume Analyzer AI

> An intelligent resume analysis tool built with **spaCy**, **scikit-learn**, and **LLM APIs** (Groq / OpenAI). Upload a PDF resume, paste a job description, and get a detailed match score + AI-powered improvement suggestions — all in a clean Streamlit web UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![spaCy](https://img.shields.io/badge/spaCy-3.7+-09a3d5?logo=spacy)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 PDF Upload | Clean text extraction via PyMuPDF |
| 🧠 NLP Extraction | Skills, education, experience via spaCy + regex |
| 📊 Match Score | TF-IDF cosine similarity between resume & JD |
| 🔑 Keyword Gap | Identifies JD keywords missing from your resume |
| 🤖 AI Feedback | Strengths, missing skills, suggestions via Groq/OpenAI |
| 🎨 Modern UI | Dark-themed, responsive Streamlit interface |

---

## 🗂 Project Structure

```
resume-analyzer/
│
├── app.py            # Streamlit UI + orchestration logic
├── analyzer.py       # LLM integration (OpenAI / Groq)
├── utils.py          # PDF parsing, NLP extraction, TF-IDF scoring
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── README.md         # This file
```

---

## 🚀 Quick Start

### 1. Clone & enter the project
```bash
git clone https://github.com/yourname/resume-analyzer.git
cd resume-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### 5. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY or OPENAI_API_KEY
```

> 💡 **Groq is recommended** — it's free, fast, and works with the same OpenAI SDK.
> Get a free key at [console.groq.com](https://console.groq.com).

### 6. Run the app
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` 🎉

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
# Use Groq (free, recommended)
GROQ_API_KEY=your_groq_api_key_here

# OR use OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

The app checks `GROQ_API_KEY` first; if not found, falls back to `OPENAI_API_KEY`.

> ⚠️ The app still works without an API key — NLP extraction and match scoring are fully offline. Only the AI feedback section requires a key.

---

## 🧠 How It Works

### Step 1 — PDF Parsing
`PyMuPDF` (`fitz`) extracts clean text from each page of the uploaded PDF.

### Step 2 — NLP Extraction (`utils.py`)
- **Skills**: Matched against a 70+ item curated keyword database using regex.
- **Education**: Detected via degree-name regex patterns + spaCy `ORG` entities.
- **Experience**: Identified via year-range patterns (`2019–2022`) + spaCy NER.
- **Contact**: Email and phone extracted via regex.

### Step 3 — Similarity Scoring
- Both texts are vectorized with `TfidfVectorizer` (unigrams + bigrams, English stop words removed).
- Cosine similarity is computed between the two vectors and expressed as a percentage.

### Step 4 — AI Feedback (`analyzer.py`)
- A structured prompt is sent to the LLM with resume text, JD text, parsed skills, match score, and missing keywords.
- The model returns a JSON object with: `strengths`, `missing_skills`, `suggestions`, `overall_summary`.

---

## 🛠 Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| NLP | spaCy (`en_core_web_sm`) |
| ML Similarity | scikit-learn (TF-IDF + cosine) |
| LLM | OpenAI API / Groq API |
| Config | python-dotenv |

---

## 📸 Screenshots

> _Upload resume → Paste JD → Click Analyze → Instant results_

The UI includes:
- Hero header with gradient background
- File uploader + text area (side by side)
- Match score ring with color-coded verdict
- Extracted skills/education/experience cards
- Missing keyword tags
- AI feedback in three columns (Strengths / Missing Skills / Suggestions)

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit your changes
4. Push and open a Pull Request

---

## 📄 License

MIT © 2024 — Free to use, modify, and distribute.
