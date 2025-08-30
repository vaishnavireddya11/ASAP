import yt_dlp
from faster_whisper import WhisperModel
from PyPDF2 import PdfReader
import docx
import os

# ----------- Audio Download -----------
def download_audio_from_url(url: str, output_path: str = "audio.mp3") -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path


# ----------- Faster Whisper Transcription -----------
def transcribe_audio_fast(
    file_path: str,
    model_size="base.en",
    device="cpu",
    compute_type="int8",
    chunk_seconds=None,
) -> str:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _ = model.transcribe(file_path)

    transcript = []
    for segment in segments:
        transcript.append(segment.text)

    return " ".join(transcript)


# ----------- File Upload Handling -----------
def extract_text_from_uploaded_file(uploaded_file) -> str:
    text = ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext == ".txt":
        text = uploaded_file.read().decode("utf-8")
    elif ext == ".pdf":
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    elif ext == ".docx":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = ""

    return text
