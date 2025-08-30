import streamlit as st
from transcript_utils import (
    download_audio_from_url,
    transcribe_audio_fast,
    extract_text_from_uploaded_file,
)
from qa_utils import (
    get_groq_client,
    answer_question,
    generate_study_plan,
    generate_quiz_json,
)

st.set_page_config(page_title="🎓 Smart Study Assistant", layout="wide")
st.title("🎓 Smart Study Assistant — Video/Transcript Q&A + Study Planner + Quiz")

# --------- Secrets / Client ----------
if "groq_client" not in st.session_state:
    api_key = st.secrets.get("groq_api_key") or st.secrets.get("general", {}).get("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ API key in .streamlit/secrets.toml")
    else:
        st.session_state.groq_client = get_groq_client(api_key)

client = st.session_state.get("groq_client")

# --------- State containers ----------
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("qa_history", [])     # list of {q, a}
st.session_state.setdefault("plans", [])          # list of plan strings
st.session_state.setdefault("quiz", [])           # list of questions dicts
st.session_state.setdefault("quiz_answers", {})   # idx -> selected option index
st.session_state.setdefault("source_label", "")   # "Video URL" / "Uploaded transcript"

# --------- Sidebar: Source selection ----------
st.sidebar.header("📥 Input Source")
source = st.sidebar.radio("Choose input type", ["Video URL", "Uploaded transcript (txt/pdf/docx)"])

if source == "Video URL":
    st.session_state["source_label"] = "Video URL"
    url = st.sidebar.text_input("Paste lecture video URL")
    fast_mode = st.sidebar.checkbox("Speed mode (chunking 60s)", value=False)
    if st.sidebar.button("Transcribe video", disabled=(client is None or not url)):
        with st.spinner("Downloading audio..."):
            path = download_audio_from_url(url)
        with st.spinner("Transcribing (optimized)..."):
            text = transcribe_audio_fast(
                path,
                model_size="base.en",
                device="cpu",          # set to "cuda" if you have GPU
                compute_type="int8",
                chunk_seconds=60 if fast_mode else None,
            )
        st.session_state["transcript"] = text
        st.success("Transcript ready ✅")

else:
    st.session_state["source_label"] = "Uploaded transcript"
    upl = st.sidebar.file_uploader("Upload transcript file", type=["txt","pdf","docx"])
    if upl and st.sidebar.button("Ingest transcript"):
        with st.spinner("Reading file..."):
            text = extract_text_from_uploaded_file(upl)
        if text.strip():
            st.session_state["transcript"] = text
            st.success("Transcript loaded ✅")
        else:
            st.error("Could not read text from the uploaded file.")

# Show a snippet of current transcript
if st.session_state["transcript"]:
    with st.expander("📄 Transcript (preview)"):
        st.write(st.session_state["transcript"][:4000] + ("..." if len(st.session_state["transcript"]) > 4000 else ""))

# --------- Tabs for features ----------
tab_qna, tab_plan, tab_quiz, tab_history = st.tabs(["💬 Q&A", "🗓 Study Plan", "📝 Quiz (MCQ)", "📚 History"])

# ---------- Q&A ----------
with tab_qna:
    st.subheader("Ask a question about the lecture")
    q = st.text_input("Your question")
    if st.button("Get Answer", disabled=(client is None or not st.session_state["transcript"] or not q)):
        with st.spinner("Thinking..."):
            ans = answer_question(client, st.session_state["transcript"], q)
        st.session_state["qa_history"].append({"question": q, "answer": ans})
        st.success("Answer added to history ✅")
        st.write(ans)

    st.markdown("---")
    st.caption("Previous Q&A (persisted):")
    for i, item in enumerate(st.session_state["qa_history"]):
        st.markdown(f"**Q{i+1}.** {item['question']}")
        st.markdown(f"**A{i+1}.** {item['answer']}")
        st.markdown("---")

# ---------- Study Plan ----------
with tab_plan:
    st.subheader("Generate a personalized study plan")
    time_text = st.text_input("Available time (e.g., '2 hours', '3 days')")
    if st.button("Generate Plan", disabled=(client is None or not st.session_state["transcript"] or not time_text)):
        with st.spinner("Planning..."):
            plan = generate_study_plan(client, st.session_state["transcript"], time_text)
        st.session_state["plans"].append(plan)
        st.success("Plan added to history ✅")
        st.write(plan)

    if st.session_state["plans"]:
        st.markdown("---")
        st.caption("Previous plans:")
        for i, p in enumerate(st.session_state["plans"]):
            with st.expander(f"Plan #{i+1}"):
                st.write(p)

# ---------- Quiz (interactive MCQ) ----------
with tab_quiz:
    st.subheader("Generate interactive MCQs")
    num_q = st.number_input("Number of questions", 1, 15, 5)
    if st.button("Create Quiz", disabled=(client is None or not st.session_state["transcript"])):
        with st.spinner("Generating quiz..."):
            quiz = generate_quiz_json(client, st.session_state["transcript"], num_q=num_q)
        st.session_state["quiz"] = quiz
        st.session_state["quiz_answers"] = {}  # reset selections
        st.success("Quiz ready ✅")

    if st.session_state["quiz"]:
        st.markdown("### Answer the questions")
        for idx, q in enumerate(st.session_state["quiz"]):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            # keep user's previous choice if any
            default = st.session_state["quiz_answers"].get(idx, 0)
            choice = st.radio(
                f"Choose an option for Q{idx+1}",
                q["options"],
                index=default if default < len(q["options"]) else 0,
                key=f"q_{idx}"
            )
            # map back to index
            st.session_state["quiz_answers"][idx] = q["options"].index(choice)

        if st.button("Submit Quiz"):
            correct = 0
            results = []
            for idx, q in enumerate(st.session_state["quiz"]):
                sel = st.session_state["quiz_answers"].get(idx, -1)
                is_ok = sel == q["correct_index"]
                correct += int(is_ok)
                results.append((idx, is_ok, q["correct_index"], sel))

            st.markdown("---")
            st.subheader(f"Score: {correct} / {len(st.session_state['quiz'])}")
            for idx, is_ok, correct_idx, sel_idx in results:
                opts = st.session_state["quiz"][idx]["options"]
                if is_ok:
                    st.success(f"Q{idx+1}: Correct ✅  ({opts[correct_idx]})")
                else:
                    chosen = opts[sel_idx] if 0 <= sel_idx < len(opts) else "No selection"
                    st.error(f"Q{idx+1}: Wrong ❌  Your answer: {chosen} | Correct: {opts[correct_idx]}")

# ---------- History (consolidated) ----------
with tab_history:
    st.subheader("All Q&A")
    if not st.session_state["qa_history"]:
        st.info("No Q&A yet.")
    else:
        for i, item in enumerate(st.session_state["qa_history"]):
            st.markdown(f"**Q{i+1}.** {item['question']}")
            st.markdown(f"**A{i+1}.** {item['answer']}")
            st.markdown("---")

    st.subheader("All Study Plans")
    if not st.session_state["plans"]:
        st.info("No study plans yet.")
    else:
        for i, p in enumerate(st.session_state["plans"]):
            with st.expander(f"Plan #{i+1}"):
                st.write(p)
