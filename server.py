"""Hands-free voice interview: FastAPI backend + a single-page browser client.

Run:  python -m uvicorn server:app --port 8000   (or double-click run.bat)
The browser plays Priya's voice, starts listening automatically when she stops,
and sends your answer automatically when you go quiet.
"""
import base64
import random
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from coach import llm
from coach.data import BASE_DIR, load_questions, load_resume_text, save_results
from coach.interviewer import InterviewState, evaluate, opening, pick_questions, respond, skip, summary
from coach.voice import speak

app = FastAPI(title="AI Interview Coach")
app.mount("/static", StaticFiles(directory=f"{BASE_DIR}/static"), name="static")

SESSIONS: dict[str, InterviewState] = {}
REPORTS: dict[str, dict] = {}


def _reply(sid: str, text: str, transcript: str | None = None) -> dict:
    state = SESSIONS[sid]
    audio = speak(text)
    return {
        "session": sid,
        "text": text,
        "transcript": transcript,
        "audio": base64.b64encode(audio).decode() if audio else None,
        "finished": state.finished,
        "progress": {"current": min(state.idx + 1, len(state.questions)), "total": len(state.questions)},
    }


def _state(sid: str) -> InterviewState:
    if sid not in SESSIONS:
        raise HTTPException(404, "Interview session not found. Please start again.")
    return SESSIONS[sid]


@app.get("/")
def index():
    return FileResponse(f"{BASE_DIR}/static/index.html")


@app.get("/api/config")
def config():
    df = load_questions()
    return {"has_key": llm.has_key(), "topics": sorted(df["Topic"].unique().tolist()), "total": len(df)}


class StartReq(BaseModel):
    name: str = "Sunil"
    topics: list[str] = []
    n: int = 8
    followups: int = 1
    intro: bool = True


@app.post("/api/start")
def start(req: StartReq):
    if not llm.has_key():
        raise HTTPException(400, "GROQ_API_KEY is missing in .env")
    qs = pick_questions(load_questions(), req.topics or None, max(1, req.n), True, req.intro)
    sid = uuid.uuid4().hex
    SESSIONS[sid] = InterviewState(questions=qs, max_followups=max(0, min(2, req.followups)))
    return _reply(sid, opening(SESSIONS[sid], req.name.strip() or "there"))


@app.post("/api/answer")
def answer(session: str = Form(...), audio: UploadFile = File(...)):
    state = _state(session)
    data = audio.file.read()
    text = llm.transcribe(data, audio.filename or "answer.webm") if len(data) > 2000 else ""
    if not text.strip():
        # Nothing real was heard: don't count it as an answer, just ask again politely.
        nudge = random.choice([
            "Sorry, I didn't quite catch that. Could you say it again?",
            "I think your audio dropped for a second. Could you repeat your answer?",
            "Sorry, I missed that. Please go ahead.",
        ])
        out = _reply(session, nudge, transcript="(couldn't hear you)")
        out["heard_nothing"] = True
        return out
    say = respond(state, text)
    return _reply(session, say, transcript=text)


class SessionReq(BaseModel):
    session: str


@app.post("/api/skip")
def skip_q(req: SessionReq):
    state = _state(req.session)
    return _reply(req.session, skip(state), transcript="(skipped)")


@app.post("/api/end")
def end(req: SessionReq):
    _state(req.session).finished = True
    return {"ok": True}


@app.get("/api/report")
def report(session: str):
    if session in REPORTS:
        return REPORTS[session]
    state = _state(session)
    results = evaluate(state, load_resume_text())
    overall, text = summary(results)
    path = save_results(results) if results else None
    REPORTS[session] = {"results": results, "overall": overall, "summary": text, "saved_to": path}
    return REPORTS[session]
