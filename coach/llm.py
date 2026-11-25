"""Thin wrapper around the Groq API (chat + JSON + speech-to-text)."""
import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# gpt-oss models think before answering; keep it short so replies stay fast.
_EXTRA = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")

_client = None


def has_key() -> bool:
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key) and key != "your_groq_api_key_here"


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def chat(messages, temperature=0.6, max_tokens=1500) -> str:
    resp = client().chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens, **_EXTRA
    )
    return resp.choices[0].message.content.strip()


def chat_json(messages, temperature=0.3, max_tokens=2000) -> dict:
    """Ask for a JSON object; fall back to extracting the first {...} block."""
    resp = client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max(max_tokens, 2000),
        response_format={"type": "json_object"},
        **_EXTRA,
    )
    text = resp.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group(0)) if match else {}


# Phrases Whisper invents when it hears silence or background noise.
GHOST_PHRASES = {
    "thankyou", "thankyouverymuch", "thanks", "thanksforwatching", "thankyouforwatching", "you", "bye",
    "byebye", "okay", "ok", "hmm", "um", "uh", "so", "pleasesubscribe", "subtitlesbytheamaraorgcommunity",
}


def is_ghost(text: str) -> bool:
    norm = "".join(ch for ch in text.lower() if ch.isalnum())
    return len(norm) < 3 or norm in GHOST_PHRASES


def transcribe(audio_bytes: bytes, filename: str = "answer.wav") -> str:
    """Speech-to-text. Drops segments Whisper itself marks as probably-not-speech."""
    resp = client().audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=WHISPER_MODEL,
        language="en",
        response_format="verbose_json",
        prompt="Technical interview answer about Python, LangChain, LangGraph, RAG, "
        "vector databases, FAISS, HNSW, embeddings, FastAPI, Docker, Kubernetes, MCP.",
    )
    segments = getattr(resp, "segments", None) or []
    if segments:
        get = lambda s, k: s.get(k) if isinstance(s, dict) else getattr(s, k, None)  # noqa: E731
        kept = [get(s, "text") or "" for s in segments if (get(s, "no_speech_prob") or 0) < 0.6]
        text = " ".join(t.strip() for t in kept).strip()
    else:
        text = (getattr(resp, "text", "") or "").strip()
    return "" if is_ghost(text) else text
