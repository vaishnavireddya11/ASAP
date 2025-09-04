from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
import faiss

# ---------------- Embedding Setup ----------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text: str) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# ---------------- FAISS Index ----------------
class PDFIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.embeddings = [get_embedding(c) for c in chunks]
        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def search(self, query, top_k=2):
        query_emb = get_embedding(query).reshape(1, -1)
        _, indices = self.index.search(query_emb, top_k)
        return [self.chunks[i] for i in indices[0]]

# ---------------- QA Pipeline ----------------
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

def answer_question(query, pdf_index: PDFIndex, top_k=2):
    retrieved_texts = pdf_index.search(query, top_k=top_k)
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = qa_pipeline(prompt, max_length=150, do_sample=False)
    return result[0]["generated_text"]

# ---------------- Study Plan ----------------
def generate_study_plan(pdf_text, available_time: str):
    prompt = f"Based on the following material:\n{pdf_text[:2000]}\n\nCreate a study plan for {available_time}."
    result = qa_pipeline(prompt, max_length=300, do_sample=False)
    return result[0]["generated_text"]

# ---------------- Quiz Generator ----------------
def generate_quiz(pdf_text, num_questions=5):
    prompt = (
        f"Based on this content:\n{pdf_text[:2000]}\n\n"
        f"Generate {num_questions} multiple-choice questions with 4 options each. "
        f"Also mention the correct option clearly."
    )
    result = qa_pipeline(prompt, max_length=500, do_sample=False)
    return result[0]["generated_text"]
