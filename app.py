"""AI Interview Coach - a human-like mock interviewer with voice and scoring.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from coach import llm
from coach.data import QUESTIONS_FILE, load_questions, load_resume_text, save_results
from coach.interviewer import INTERVIEWER_NAME, InterviewState, evaluate, opening, pick_questions, respond, skip, summary
from coach.voice import speak

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="centered")

ss = st.session_state
ss.setdefault("state", None)
ss.setdefault("audio_turn", -1)  # index of the last interviewer turn already spoken
ss.setdefault("rec_n", 0)  # changes the recorder key so it resets after each answer
ss.setdefault("draft", "")
ss.setdefault("report", None)

st.title("🎤 AI Interview Coach")
st.caption(f"Mock technical interview with {INTERVIEWER_NAME}. Answer by voice or text; get a score out of 10 at the end.")

if not llm.has_key():
    st.error("GROQ_API_KEY is missing. Copy `.env.example` to `.env`, paste your Groq key, then restart the app.")
    st.stop()


@st.cache_data
def _questions():
    return load_questions()


@st.cache_data
def _resume():
    return load_resume_text()


try:
    qdf = _questions()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read {QUESTIONS_FILE}: {e}")
    st.stop()

# ---------------- Sidebar: setup ----------------
with st.sidebar:
    st.header("Interview setup")
    disabled = ss.state is not None and not ss.state.finished
    name = st.text_input("Your name", "Sunil", disabled=disabled)
    topics = st.multiselect("Topics (empty = all)", sorted(qdf["Topic"].unique()), disabled=disabled)
    pool = qdf if not topics else qdf[qdf["Topic"].isin(topics)]
    n_q = st.slider("Number of questions", 1, max(1, len(pool)), min(8, len(pool)), disabled=disabled)
    shuffle = st.checkbox("Random order", True, disabled=disabled)
    intro = st.checkbox("Start with 'Tell me about yourself'", True, disabled=disabled)
    max_fu = st.select_slider("Follow-up questions per topic", [0, 1, 2], 1, disabled=disabled)
    mode = st.radio("How will you answer?", ["Voice 🎙️", "Text ⌨️"], disabled=disabled)
    tts_on = st.checkbox(f"{INTERVIEWER_NAME} speaks out loud", True)
    st.caption(f"{len(qdf)} questions loaded from `{QUESTIONS_FILE}`")

    if st.button("▶️ Start interview", type="primary", disabled=disabled, use_container_width=True):
        state = InterviewState(
            questions=pick_questions(qdf, topics, n_q, shuffle, intro), max_followups=max_fu
        )
        with st.spinner(f"{INTERVIEWER_NAME} is joining the call..."):
            opening(state, name)
        ss.state, ss.report, ss.audio_turn, ss.draft = state, None, -1, ""
        ss.mode = mode
        st.rerun()

    if disabled and st.button("⏹️ End interview now", use_container_width=True):
        ss.state.finished = True
        st.rerun()

state: InterviewState | None = ss.state
if state is None:
    st.info("Choose your settings in the sidebar and press **Start interview**. "
            "Speak as you would in a real interview: the interviewer may ask follow-ups.")
    st.stop()

# ---------------- Conversation ----------------
total = len(state.questions)
if not state.finished:
    st.progress(min(state.idx, total) / total, text=f"Question {min(state.idx + 1, total)} of {total}")

last_interviewer = max((k for k, t in enumerate(state.turns) if t.role == "interviewer"), default=-1)
for k, t in enumerate(state.turns):
    if t.role == "interviewer":
        with st.chat_message("assistant", avatar="👩‍💼"):
            st.write(t.text)
            if tts_on and k == last_interviewer and ss.audio_turn != k:
                audio = speak(t.text)
                if audio:
                    st.audio(audio, format="audio/mp3", autoplay=True)
                ss.audio_turn = k
    else:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(t.text)


def _send(answer: str):
    with st.spinner(f"{INTERVIEWER_NAME} is thinking..."):
        respond(state, answer)
    ss.draft = ""
    ss.rec_n += 1
    st.rerun()


# ---------------- Answer input ----------------
if not state.finished:
    if ss.get("mode", "Voice 🎙️").startswith("Voice"):
        rec = st.audio_input("Record your answer, then stop", key=f"rec_{ss.rec_n}")
        if rec is not None and not ss.draft:
            with st.spinner("Transcribing..."):
                ss.draft = llm.transcribe(rec.getvalue(), "answer.wav")
        if ss.draft:
            edited = st.text_area("Your answer (fix any transcription mistakes)", ss.draft, height=140)
            c1, c2 = st.columns(2)
            if c1.button("✅ Send answer", type="primary", use_container_width=True):
                _send(edited)
            if c2.button("🔁 Record again", use_container_width=True):
                ss.draft = ""
                ss.rec_n += 1
                st.rerun()
    typed = st.chat_input("Type your answer here...")
    if typed:
        _send(typed)
    if st.button("⏭️ Skip this question"):
        with st.spinner("Moving on..."):
            skip(state)
        ss.rec_n += 1
        st.rerun()
    st.stop()

# ---------------- Report ----------------
if ss.report is None:
    with st.spinner("Evaluating your answers..."):
        results = evaluate(state, _resume())
        overall, text = summary(results)
        path = save_results(results) if results else None
    ss.report = {"results": results, "overall": overall, "summary": text, "path": path}

rep = ss.report
st.divider()
st.header("📊 Your interview report")
if not rep["results"]:
    st.warning("No answers were given, so there is nothing to score.")
    st.stop()

res = pd.DataFrame(rep["results"])
c1, c2, c3 = st.columns(3)
c1.metric("Overall score", f"{rep['overall']} / 10")
c2.metric("Questions answered", len(res))
c3.metric("Best topic", res.groupby("Topic")["Score"].mean().idxmax())
st.write(rep["summary"])

st.subheader("Score by topic")
st.bar_chart(res.groupby("Topic")["Score"].mean().sort_values())

st.subheader("Question by question")
for r in rep["results"]:
    with st.expander(f"{r['Score']}/10 · [{r['Topic']}] {r['Question']}"):
        st.markdown(f"**Verdict:** {r['Verdict']}")
        st.markdown(f"**Good:** {r['Strengths']}")
        st.markdown(f"**Missed:** {r['Missed']}")
        st.markdown(f"**A stronger answer:** {r['Better answer']}")

st.download_button("⬇️ Download this report (CSV)", res.to_csv(index=False).encode("utf-8"),
                   "interview_report.csv", "text/csv")
if rep["path"]:
    st.caption(f"Saved to `{rep['path']}` so you can track progress over time.")
if st.button("🔄 Start a new interview", type="primary"):
    ss.state, ss.report = None, None
    st.rerun()
