"""Interview flow: a human-like interviewer that asks, probes with follow-ups,
then an evaluator that scores each answer out of 10 at the end."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from . import llm

INTERVIEWER_NAME = "Priya"

PERSONA = f"""You are {INTERVIEWER_NAME}, a senior AI engineering interviewer at an Indian tech company.
You are interviewing a candidate for an AI Engineer / GenAI Engineer role.
Behave like a real human interviewer on a video call:
- Speak naturally and briefly (1-3 short sentences). Plain spoken English, no markdown, no bullet points, no emojis.
- Ask exactly ONE question at a time.
- Never reveal scores, never say whether an answer was right or wrong, never teach the answer during the interview.
- React with short neutral acknowledgements ("Okay.", "Got it.", "Makes sense.", "Alright.") and vary them.
- You may rephrase the planned question in your own words, but keep its technical meaning.
- If the candidate mentions a specific tool, project or design choice, it is natural to probe it.
"""

DECIDE = """Planned question for this part of the interview:
"{question}"   (topic: {topic})

Conversation on this question so far:
{transcript}

Follow-ups already asked on this question: {used} of max {max_fu}.
Clarifications already given on this question: {clar} of max 2.
Next planned question: {next_q}

Your last few lines in this interview (do NOT start with the same words again, vary your reactions):
{recent}

Look ONLY at the candidate's latest message and decide your next move. Reply ONLY with JSON:
{{"decision": "repeat" | "clarify" | "wait" | "follow_up" | "move_on",
  "say": "<exactly what you say out loud>",
  "question": "<only the question sentence contained in 'say', or empty>"}}

Rules:
- "repeat": the candidate asked you to repeat / say it again / didn't hear you. (The app will repeat your last question word for word, so 'say' can be empty.)
- "clarify": the candidate says they didn't understand the question or asks what you mean. Explain it more simply with a tiny hint of context, without giving the answer. Only if clarifications remain, otherwise "move_on".
- "wait": the candidate asks for a moment to think ("let me think", "one second"). Say something short like "Sure, take your time." Do not ask anything new.
- "follow_up": only if follow-ups remain AND the answer was vague, too short, missed an obvious key point, or mentioned something worth probing (a tool, trade-off, their project). Ask one sharp follow-up like a real interviewer would.
- "move_on": the answer is reasonably complete, the candidate said they don't know / wants to skip, the latest message is not a real answer and nothing above fits, or no follow-ups remain. Say a brief neutral acknowledgement, then a natural transition, then ask the next planned question. If the next planned question is NONE, thank them warmly, say the interview is complete and that you'll share feedback now.
- Never repeat a sentence you already said, except for "repeat".
"""

EVALUATE = """You are a strict but fair technical interview evaluator for an AI Engineer role in India.
Score the candidate's performance on ONE interview question, using the whole exchange (main answer plus follow-ups).
The answers came from speech-to-text, so ignore small transcription mistakes and filler words.

Candidate resume (for context on their real projects):
{resume}

Question (topic: {topic}): {question}

Exchange:
{transcript}

Scoring rubric (total 10):
- Technical correctness: 0-4
- Depth / key concepts covered: 0-3
- Practical example or link to their own project: 0-2
- Clarity and structure: 0-1
A "don't know" or empty answer scores 0-1.

Reply ONLY with JSON:
{{"score": <integer 0-10>,
  "verdict": "<one short sentence>",
  "strengths": "<what was good, 1-2 short points>",
  "missed": "<key points that were missing or wrong, 1-3 short points>",
  "better_answer": "<a model spoken answer of 60-110 words the candidate could give, mentioning their own projects where relevant>"}}
"""

SUMMARY = """You are an interview coach. Here are per-question results of a mock interview for an AI Engineer role:
{results}

Overall score: {overall}/10.
Write a short, encouraging but honest summary for the candidate in simple English (max 120 words):
1) overall impression, 2) top 3 things to improve before the next real interview, 3) one strength to keep using.
No markdown headings; use plain sentences or a short numbered list."""


@dataclass
class Turn:
    role: str  # "interviewer" | "candidate"
    text: str
    q_index: int


@dataclass
class InterviewState:
    questions: list[dict]  # [{"topic":..., "question":...}]
    max_followups: int = 1
    idx: int = 0
    followups_used: int = 0
    turns: list[Turn] = field(default_factory=list)
    finished: bool = False
    last_question: str = ""  # the exact question sentence Priya asked most recently
    clarify_used: int = 0  # repeats/clarifications on the current question

    def to_dict(self):
        return asdict(self)


def pick_questions(df, topics=None, n=8, shuffle=True, include_intro=True) -> list[dict]:
    rows = df if not topics else df[df["Topic"].isin(topics)]
    items = [{"topic": r.Topic, "question": r.Question} for r in rows.itertuples()]
    if shuffle:
        random.shuffle(items)
    items = items[:n]
    if include_intro:
        items.insert(0, {"topic": "Introduction", "question": "Tell me about yourself and your current role."})
    return items


def _transcript(state: InterviewState, q_index: int) -> str:
    lines = []
    for t in state.turns:
        if t.q_index == q_index:
            who = "Interviewer" if t.role == "interviewer" else "Candidate"
            lines.append(f"{who}: {t.text}")
    return "\n".join(lines) or "(nothing yet)"


def opening(state: InterviewState, candidate_name: str) -> str:
    first = state.questions[0]
    msgs = [
        {"role": "system", "content": PERSONA},
        {
            "role": "user",
            "content": f"Start the interview. Greet {candidate_name} warmly in one sentence, introduce yourself "
            f"as {INTERVIEWER_NAME} from the hiring team, say this will be a technical round of about "
            f"{len(state.questions)} questions, then ask the first question: \"{first['question']}\". "
            'Reply ONLY with JSON: {"say": "<what you say out loud>", "question": "<only the question sentence>"}',
        },
    ]
    out = llm.chat_json(msgs, temperature=0.6)
    say = (out.get("say") or "").strip() or f"Hi {candidate_name}, I'm {INTERVIEWER_NAME}. {first['question']}"
    state.last_question = (out.get("question") or "").strip() or first["question"]
    if _norm(state.last_question)[:30] not in _norm(say):  # model forgot to actually ask it
        say = f"{say} {state.last_question}"
    state.turns.append(Turn("interviewer", say, 0))
    return say


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _interviewer_lines(state: InterviewState, k: int = 4) -> list[str]:
    return [t.text for t in state.turns if t.role == "interviewer"][-k:]


def respond(state: InterviewState, answer: str) -> str:
    """Record the candidate's answer and return what the interviewer says next."""
    i = state.idx
    state.turns.append(Turn("candidate", answer.strip() or "(no answer)", i))
    q = state.questions[i]
    nxt = state.questions[i + 1]["question"] if i + 1 < len(state.questions) else "NONE"
    msgs = [
        {"role": "system", "content": PERSONA},
        {
            "role": "user",
            "content": DECIDE.format(
                question=q["question"],
                topic=q["topic"],
                transcript=_transcript(state, i),
                used=state.followups_used,
                max_fu=state.max_followups,
                clar=state.clarify_used,
                next_q=nxt,
                recent="\n".join(f"- {line}" for line in _interviewer_lines(state)) or "(none)",
            ),
        },
    ]
    out = llm.chat_json(msgs, temperature=0.7)
    decision = out.get("decision", "move_on")
    say = (out.get("say") or "").strip()
    question = (out.get("question") or "").strip()
    end_line = "Thank you, that's the end of our interview. I'll share your feedback now."
    plain_next = f"Alright, let's move on. {nxt}" if nxt != "NONE" else end_line

    # Guard rails so the conversation can never get stuck.
    if decision in ("repeat", "clarify") and state.clarify_used >= 2:
        decision, say, question = "move_on", plain_next, (nxt if nxt != "NONE" else "")
    if decision == "follow_up" and state.followups_used >= state.max_followups:
        decision, say, question = "move_on", plain_next, (nxt if nxt != "NONE" else "")
    if decision not in ("repeat", "wait") and (
        not say or any(_norm(say) == _norm(prev) for prev in _interviewer_lines(state, 6))
    ):
        decision, say, question = "move_on", plain_next, (nxt if nxt != "NONE" else "")

    if decision == "repeat":
        state.clarify_used += 1
        say = f"Sure, I'll repeat it. {state.last_question}"
        state.turns.append(Turn("interviewer", say, i))
    elif decision == "wait":
        say = say or "Sure, take your time."
        state.turns.append(Turn("interviewer", say, i))
    elif decision == "clarify":
        state.clarify_used += 1
        state.last_question = question or state.last_question
        state.turns.append(Turn("interviewer", say, i))
    elif decision == "follow_up":
        state.followups_used += 1
        state.last_question = question or say
        state.turns.append(Turn("interviewer", say, i))
    else:
        state.idx += 1
        state.followups_used = 0
        state.clarify_used = 0
        state.last_question = question or (nxt if nxt != "NONE" else "")
        if state.idx >= len(state.questions):
            state.finished = True
            state.turns.append(Turn("interviewer", say, i))
        else:
            state.turns.append(Turn("interviewer", say, state.idx))
    return say


def skip(state: InterviewState) -> str:
    return respond(state, "I'm not sure about this one, can we move to the next question?")


def evaluate(state: InterviewState, resume: str) -> list[dict]:
    results = []
    for i, q in enumerate(state.questions):
        tr = _transcript(state, i)
        if "Candidate:" not in tr:
            continue
        out = llm.chat_json(
            [
                {
                    "role": "user",
                    "content": EVALUATE.format(
                        resume=resume or "(not provided)", topic=q["topic"], question=q["question"], transcript=tr
                    ),
                }
            ],
            temperature=0.2,
            max_tokens=700,
        )
        try:
            score = max(0, min(10, int(round(float(out.get("score", 0))))))
        except (TypeError, ValueError):
            score = 0
        results.append(
            {
                "Topic": q["topic"],
                "Question": q["question"],
                "Score": score,
                "Verdict": out.get("verdict", ""),
                "Strengths": out.get("strengths", ""),
                "Missed": out.get("missed", ""),
                "Better answer": out.get("better_answer", ""),
            }
        )
    return results


def summary(results: list[dict]) -> tuple[float, str]:
    if not results:
        return 0.0, "No answers to evaluate."
    overall = round(sum(r["Score"] for r in results) / len(results), 1)
    brief = "\n".join(f"- [{r['Topic']}] {r['Question']} -> {r['Score']}/10. Missed: {r['Missed']}" for r in results)
    text = llm.chat([{"role": "user", "content": SUMMARY.format(results=brief, overall=overall)}], temperature=0.5)
    return overall, text
