# 🎤 AI Interview Coach

A mock technical interviewer that talks like a real person. It asks your real interview questions one by one, listens to your spoken answer, asks natural follow-ups ("Why HNSW and not IVF?"), and at the end scores every answer out of 10 with feedback and a stronger model answer.

Built with **Python, FastAPI, Groq (gpt-oss-120b + Whisper large-v3-turbo)**, **Edge neural TTS** and browser voice-activity detection.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat-square)
![Whisper](https://img.shields.io/badge/Whisper-412991?style=flat-square)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)

## Features
- **Your own question bank**: reads `data/Interview_Questions.xlsx` (sheet `Interview Questions`, columns `Topic`, `Question`). Add new questions any time.
- **Human-like interviewer ("Priya")**: greets you, rephrases questions naturally, gives neutral reactions, never reveals answers mid-interview, probes vague answers with follow-ups, and rephrases if you ask her to repeat.
- **Voice in, voice out**: record your answer in the browser; Whisper transcribes it (you can fix the text before sending). The interviewer speaks with an Indian-English neural voice.
- **Resume-aware**: reads `data/resume.pdf` so feedback and model answers reference your real projects.
- **Scored report**: per-question score /10 (correctness, depth, examples, clarity), what you missed, a model answer, score by topic, overall summary, and a history file in `results/practice_history.xlsx`.

## How it works
```
Excel questions ─┐
Resume (PDF) ────┼─> Interviewer LLM ──(decide: follow_up / clarify / move_on)──> speaks next line
Your voice ─> Whisper ─> transcript ─┘                     │
                                                           ▼
                                   Evaluator LLM (rubric /10 per question) ─> Report + history
```

## Setup (Windows)
```powershell
cd D:\Projects\AI-Interview-Coach
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
notepad .env        # paste your key: GROQ_API_KEY=...
run.bat             # starts the server and opens http://localhost:8000
```
Get a free API key at https://console.groq.com/keys. Never share or commit `.env`.

Allow microphone access when the browser asks. Earphones are recommended.

### Two modes
- **Hands-free voice interview (default, `server.py` + `static/index.html`)**: Priya speaks, the mic turns on by itself when she finishes, and your answer is sent automatically after a few seconds of silence (voice activity detection in the browser). Built with FastAPI.
- **Text/review mode (`app.py`)**: Streamlit UI where you can edit transcripts before sending: `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Tips
- Choose topics in the sidebar (e.g. only "Kubernetes") to drill weak areas.
- Set follow-ups to 2 for a tougher, more realistic round.
- Change the interviewer voice with `TTS_VOICE` in `.env` (e.g. `en-IN-PrabhatNeural` for a male voice).

## Roadmap
- Real-time, interruptible voice (Gemini Live / OpenAI Realtime)
- LangGraph version of the interview loop with checkpoints
- Company-specific question sets and difficulty levels
