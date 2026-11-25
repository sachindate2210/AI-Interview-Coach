"""Text-to-speech with Microsoft Edge neural voices (free, needs internet)."""
import asyncio
import os

import edge_tts

VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")


async def _speak(text: str, voice: str) -> bytes:
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def speak(text: str, voice: str = VOICE) -> bytes | None:
    """Return MP3 bytes, or None if TTS fails (the app then falls back to text)."""
    try:
        return asyncio.run(_speak(text, voice))
    except Exception:
        return None
