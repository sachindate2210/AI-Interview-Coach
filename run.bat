@echo off
cd /d "%~dp0"
if not exist .env (
  copy .env.example .env >nul
  echo Created .env - open it in Notepad and paste your GROQ_API_KEY, then run this again.
  notepad .env
  exit /b
)
echo Starting AI Interview Coach at http://localhost:8000 ...
start "" http://localhost:8000
.\.venv\Scripts\python.exe -m uvicorn server:app --port 8000
