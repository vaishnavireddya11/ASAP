import json
from typing import List, Dict, Any
from groq import Groq

def get_groq_client(groq_api_key: str) -> Groq:
    return Groq(api_key=groq_api_key)

# ---------------- Q&A ----------------
def answer_question(client: Groq, transcript: str, question: str, model: str = "llama-3.1-8b-instant") -> str:
    prompt = f"""You are an expert tutor. Answer ONLY using the context below.
If the answer isn't in the context, say "I don't have that in the lecture."

Context:
{transcript[:6000]}

Question: {question}
Answer briefly and clearly:"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
    )
    return resp.choices[0].message.content.strip()

# ------------- Study Plan -------------
def generate_study_plan(client: Groq, transcript: str, time_text: str, model: str = "llama-3.1-8b-instant") -> str:
    prompt = f"""Create a concise, efficient study plan based on this lecture content.
User has {time_text} total.
Prioritize the most important topics first, include time splits, and end with a short revision checklist.

Lecture (partial):
{transcript[:6000]}"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
    )
    return resp.choices[0].message.content.strip()

# ------------- Quiz (JSON format) -------------
def generate_quiz_json(client: Groq, transcript: str, num_q: int = 5, model: str = "llama-3.1-8b-instant") -> List[Dict[str, Any]]:
    """
    Ask model to return strict JSON: [{question, options: [..], correct_index}]
    """
    prompt = f"""Create a multiple-choice quiz based ONLY on the lecture below.
Return STRICT JSON (no prose). Format:
[
  {{"question": "...", "options": ["A","B","C","D"], "correct_index": 0}},
  ...
]

Make {num_q} questions, 1 correct + 3 plausible distractors each.
Lecture:
{transcript[:6000]}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()

    # best-effort JSON parsing
    try:
        # sometimes model wraps in code fences
        if content.startswith("```"):
            content = content.strip("`")
            # remove possible "json" after ```
            content = content.split("\n", 1)[-1]
        data = json.loads(content)
        # minimal validation
        clean = []
        for item in data:
            q = str(item.get("question","")).strip()
            opts = list(item.get("options", []))
            ci = int(item.get("correct_index", 0))
            if q and isinstance(opts, list) and len(opts) >= 2 and 0 <= ci < len(opts):
                clean.append({"question": q, "options": opts, "correct_index": ci})
        return clean[:num_q]
    except Exception:
        # fallback single question
        return [{
            "question": "Which of the following is correct according to the lecture?",
            "options": ["Option A","Option B","Option C","Option D"],
            "correct_index": 0
        }]
