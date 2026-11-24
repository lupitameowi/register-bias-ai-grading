"""
ai_professor.py: the AI grader used as a measurement instrument.

Study: Register Bias in AI Grading (G. Rodriguez, Hartnell College).

  * Analytic rubric, five dimensions scored 1-5. Comprehension and organization
    are content dimensions, defined to be register-neutral. Grammar/mechanics,
    word choice, and academic tone are style dimensions (manipulation check).
  * Holistic prompt: "how well does this summary capture the passage?" It does
    not mention tone or style.
  * The grader sees the source passage but not the key-point checklist.
  * Each text is graded in a fresh session unless an order test places other
    texts before it.
  * Each text is scored N_RUNS times and the median is used.
"""

import json
import os
import re
import statistics

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL = "claude-sonnet-5-5"   # model used for all scoring
N_RUNS = 5                    # scorings per text; the median is used
MAX_TOKENS = 2000
SCORE_MIN, SCORE_MAX = 1, 5

HERE = os.path.dirname(os.path.abspath(__file__))
MATERIALS = os.path.join(HERE, "materials")

# ----------------------------------------------------------------------------
# The rubric (Table 1 in the paper)
# ----------------------------------------------------------------------------
RUBRIC = {
    "comprehension": {
        "type": "content",
        "definition": "How accurately and completely the summary captures the passage's key ideas. "
                      "Judge the ideas only. Do NOT let wording, vocabulary level, formality, or tone affect this score.",
        1: "Misses or distorts most key ideas.",
        3: "Captures some key ideas accurately but leaves out or blurs others.",
        5: "Accurately captures all of the key ideas.",
    },
    "organization": {
        "type": "content",
        "definition": "How logically the ideas are ordered and connected. "
                      "Do NOT let formality or tone affect this score; casual writing can be well organized.",
        1: "Ideas are jumbled; no clear order.",
        3: "Mostly clear order with some jumps.",
        5: "Ideas follow a clear, logical order and connect to each other.",
    },
    "grammar_mechanics": {
        "type": "style",
        "definition": "Correctness of sentence structure, spelling, and punctuation.",
        1: "Frequent errors that make the writing hard to follow.",
        3: "Some errors, but meaning is clear.",
        5: "Few or no errors.",
    },
    "word_choice": {
        "type": "style",
        "definition": "Precision and sophistication of vocabulary.",
        1: "Vague or very limited vocabulary.",
        3: "Adequate, everyday vocabulary.",
        5: "Precise, varied, discipline-appropriate vocabulary.",
    },
    "academic_tone": {
        "type": "style",
        "definition": "How formal and academic the writing sounds.",
        1: "Casual and conversational.",
        3: "A mix of casual and formal.",
        5: "Consistently formal and academic.",
    },
}
DIMENSIONS = list(RUBRIC)
CONTENT_DIMS = [d for d in DIMENSIONS if RUBRIC[d]["type"] == "content"]
STYLE_DIMS = [d for d in DIMENSIONS if RUBRIC[d]["type"] == "style"]


def _rubric_text():
    lines = []
    for dim, spec in RUBRIC.items():
        lines.append(f"- {dim} ({spec['type']}): {spec['definition']}")
        for level in (1, 3, 5):
            lines.append(f"    {level} = {spec[level]}")
        lines.append("    (2 and 4 fall between the described levels.)")
    return "\n".join(lines)


# Doubled braces survive str.format() as literal { } in the final prompt.
_JSON_SHAPE = "{{" + ", ".join(f'"{d}": <1-5>' for d in DIMENSIONS) + "}}"

# Three wordings of the SAME analytic rubric. Variant A is the main prompt;
# B and C exist only to measure prompt sensitivity in Phase 1 (H1).
ANALYTIC_VARIANTS = {
    "A": ("You are grading a student's summary of the passage below. Score the summary "
          "on each rubric dimension using the full 1-5 scale.\n\n"
          "PASSAGE:\n{passage}\n\nRUBRIC:\n{rubric}\n\n"
          "Return ONLY a JSON object with integer scores: " + _JSON_SHAPE),
    "B": ("Below is a source passage, followed by a rubric. A student has written a summary "
          "of the passage, which you will receive next. Rate the summary from 1 (lowest) to 5 "
          "(highest) on every rubric dimension.\n\n"
          "SOURCE PASSAGE:\n{passage}\n\nSCORING RUBRIC:\n{rubric}\n\n"
          "Reply with JSON only, in this shape: " + _JSON_SHAPE),
    "C": ("Act as an instructor scoring a summary assignment. Use this rubric, scoring each "
          "dimension from 1 to 5:\n{rubric}\n\n"
          "The passage the student summarized:\n{passage}\n\n"
          "Output nothing but a JSON object: " + _JSON_SHAPE),
}

HOLISTIC_PROMPT = (
    "Here is a passage. You will then receive a student's summary of it.\n\n"
    "PASSAGE:\n{passage}\n\n"
    "On a scale from 1 (poor) to 5 (excellent), how well does the summary capture the passage? "
    'Return ONLY a JSON object: {{"holistic": <1-5>}}'
)

# Used only for the Phase 1 accuracy check on SmART explanatory responses.
EXPLANATORY_PROMPT = (
    "You will receive a student's explanatory response to this writing task:\n{passage}\n\n"
    "On a scale from 1 (poor) to 5 (excellent), rate the overall quality of the response. "
    'Return ONLY a JSON object: {{"holistic": <1-5>}}'
)


def system_prompt(passage_text, mode="analytic", variant="A"):
    if mode == "analytic":
        return ANALYTIC_VARIANTS[variant].format(passage=passage_text, rubric=_rubric_text())
    if mode == "holistic":
        return HOLISTIC_PROMPT.format(passage=passage_text)
    if mode == "explanatory":
        return EXPLANATORY_PROMPT.format(passage=passage_text)
    raise ValueError(mode)


def expected_keys(mode):
    return DIMENSIONS if mode == "analytic" else ["holistic"]


# ----------------------------------------------------------------------------
# Objective text measures (manipulation check + length control)
# ----------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z']+")
_CONTRACTION = re.compile(r"\b\w+'(s|re|ve|ll|d|t|m)\b", re.I)
SLANG = {"basically", "kinda", "gonna", "wanna", "gotta", "stuff", "okay", "ok", "cause",
         "pretty", "super", "totally", "messed", "bunch", "cool", "yeah", "lol"}
SECOND_PERSON = {"you", "your", "you're", "yours"}


def words(text):
    return _WORD.findall(text)


def word_count(text):
    return len(words(text))


def _syllables(word):
    word = word.lower().strip("'")
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1 and not word.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def flesch_kincaid_grade(text):
    ws = words(text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    if not ws:
        return 0.0
    syl = sum(_syllables(w) for w in ws)
    return round(0.39 * (len(ws) / sentences) + 11.8 * (syl / len(ws)) - 15.59, 2)


def informal_markers_per_100(text):
    ws = [w.lower() for w in words(text)]
    if not ws:
        return 0.0
    count = len(_CONTRACTION.findall(text)) + sum(w in SLANG for w in ws) + sum(w in SECOND_PERSON for w in ws)
    return round(100 * count / len(ws), 2)


def text_measures(text):
    return {"word_count": word_count(text),
            "fk_grade": flesch_kincaid_grade(text),
            "informal_per_100": informal_markers_per_100(text)}


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
def load_passages():
    with open(os.path.join(MATERIALS, "passages.json"), encoding="utf-8") as f:
        return json.load(f)


def key_point_coverage(text, passage):
    """Automatic keyword count of the passage's key points present in a text.
    Used for the Phase 1 materials check. Phase 2 checklist counts were scored by hand."""
    low = text.lower()
    return sum(any(k in low for k in kp["keywords"]) for kp in passage["key_points"])


# ----------------------------------------------------------------------------
# The grader: one "session" = one conversation. Each text is a user turn.
# A fresh session per text means no memory of earlier texts.
# ----------------------------------------------------------------------------
def _clamp(x):
    return max(SCORE_MIN, min(SCORE_MAX, int(round(x))))


def _parse(raw, mode):
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1])
    return {k: _clamp(data[k]) for k in expected_keys(mode)}


def _live_session(system, texts, mode):
    """Grade texts in order inside ONE conversation; return the scores for each."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    messages, out = [], []
    for text in texts:
        messages.append({"role": "user", "content": f"STUDENT SUMMARY:\n{text}"})
        for attempt in range(3):
            # No temperature argument: the model only accepts its default sampling.
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=messages)
            raw = "".join(b.text for b in resp.content if b.type == "text")
            try:
                scores = _parse(raw, mode)
                break
            except (ValueError, KeyError, TypeError):
                if attempt == 2:
                    raise
        # Pass the assistant turn back unmodified (keeps any thinking blocks intact).
        messages.append({"role": "assistant", "content": resp.content})
        out.append(scores)
    return out


def grade_session(passage, texts, mode="analytic", variant="A"):
    """Grade a list of texts in ONE session, in order. Returns one score dict per text."""
    system = system_prompt(passage["text"], mode, variant)
    return _live_session(system, texts, mode)


def score_text(passage, text, mode="analytic", variant="A", n_runs=N_RUNS, context=None):
    """Score one text n_runs times, each run in a fresh session.

    If `context` (a list of other texts) is given, those texts are graded first
    in the same session and only the target's score is kept (order test).
    Returns (median_scores, list_of_run_scores)."""
    context = context or []
    runs = []
    for _ in range(n_runs):
        session = grade_session(passage, context + [text], mode, variant)
        runs.append(session[-1])
    keys = expected_keys(mode)
    median = {k: statistics.median(r[k] for r in runs) for k in keys}
    return median, runs
