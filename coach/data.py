"""Load questions and resume, save results."""
import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


QUESTIONS_FILE = _p(os.getenv("QUESTIONS_FILE", "data/Interview_Questions.xlsx"))
RESUME_FILE = _p(os.getenv("RESUME_FILE", "data/resume.pdf"))
RESULTS_FILE = _p("results/practice_history.xlsx")


def load_questions(path: str = QUESTIONS_FILE) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Interview Questions")
    df.columns = [str(c).strip() for c in df.columns]
    if "Topic" not in df.columns:
        df["Topic"] = "General"
    df = df.dropna(subset=["Question"])
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Topic"] = df["Topic"].fillna("General").astype(str).str.strip()
    return df.reset_index(drop=True)


def load_resume_text(path: str = RESUME_FILE, max_chars: int = 6000) -> str:
    if not os.path.exists(path):
        return ""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text[:max_chars]


def save_results(rows: list[dict], path: str = RESULTS_FILE) -> str:
    """Append this session's per-question scores to an Excel history file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new = pd.DataFrame([{"Session": stamp, **r} for r in rows])
    if os.path.exists(path):
        new = pd.concat([pd.read_excel(path), new], ignore_index=True)
    new.to_excel(path, index=False)
    return path
