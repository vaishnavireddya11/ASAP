from PyPDF2 import PdfReader

def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extracts raw text from uploaded PDF file."""
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    """Splits text into smaller chunks for embeddings."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
