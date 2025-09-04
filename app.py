import streamlit as st
from pdf_utils import extract_text_from_uploaded_file, chunk_text
from qa_utils import PDFIndex, answer_question, generate_study_plan, generate_quiz

st.set_page_config(page_title="📘 Smart PDF Assistant", layout="wide")
st.title("📘 Smart PDF Assistant — Q&A + Study Planner + Quiz")

# --------- State containers ----------
st.session_state.setdefault("pdf_text", "")
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("index", None)
st.session_state.setdefault("qa_history", [])
st.session_state.setdefault("plans", [])
st.session_state.setdefault("quizzes", [])

# --------- Sidebar: PDF Upload ----------
st.sidebar.header("📥 Upload PDF")
upl = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])
if upl and st.sidebar.button("Process PDF"):
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_uploaded_file(upl)
    if text.strip():
        st.session_state["pdf_text"] = text
        st.session_state["chunks"] = chunk_text(text)
        st.session_state["index"] = PDFIndex(st.session_state["chunks"])
        st.success("✅ PDF processed and indexed successfully!")

# --------- Show content preview ----------
if st.session_state["pdf_text"]:
    with st.expander("📄 Document Content (preview)"):
        st.write(st.session_state["pdf_text"][:2000] + "...")

# --------- Tabs for features ----------
tab_qna, tab_plan, tab_quiz, tab_history = st.tabs(["💬 Q&A", "🗓 Study Plan", "📝 Quiz", "📚 History"])

# ---------- Q&A ----------
with tab_qna:
    st.subheader("Ask a question about the PDF")
    q = st.text_input("Your question")
    if st.button("Get Answer", disabled=(st.session_state["index"] is None or not q)):
        with st.spinner("Thinking..."):
            ans = answer_question(q, st.session_state["index"])
        st.session_state["qa_history"].append({"question": q, "answer": ans})
        st.write(ans)

# ---------- Study Plan ----------
with tab_plan:
    st.subheader("Generate a personalized study plan")
    time_text = st.text_input("Available time (e.g., '2 hours', '3 days')")
    if st.button("Generate Plan", disabled=(not st.session_state["pdf_text"] or not time_text)):
        with st.spinner("Planning..."):
            plan = generate_study_plan(st.session_state["pdf_text"], time_text)
        st.session_state["plans"].append(plan)
        st.write(plan)

# ---------- Quiz ----------
with tab_quiz:
    st.subheader("Generate Quiz Questions")
    num_q = st.number_input("Number of questions", 1, 10, 5)
    if st.button("Create Quiz", disabled=(not st.session_state["pdf_text"])):
        with st.spinner("Generating quiz..."):
            quiz = generate_quiz(st.session_state["pdf_text"], num_q)
        st.session_state["quizzes"].append(quiz)
        st.write(quiz)

# ---------- History ----------
with tab_history:
    st.subheader("All Q&A History")
    for i, item in enumerate(st.session_state["qa_history"]):
        st.markdown(f"**Q{i+1}.** {item['question']}")
        st.markdown(f"**A{i+1}.** {item['answer']}")
