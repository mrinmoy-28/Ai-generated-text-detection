# backend/file_parser.py
import os
import io

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded file.
    Supports: .txt, .pdf, .docx
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return _parse_txt(file_bytes)
    elif ext == ".pdf":
        return _parse_pdf(file_bytes)
    elif ext == ".docx":
        return _parse_docx(file_bytes)
    else:
        # Try treating as plain text anyway
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except:
            raise ValueError(f"Unsupported file type: {ext}")


def _parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def _parse_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text   = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")


def _parse_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc  = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")


def validate_text(text: str, min_words: int = 20) -> bool:
    """Check if extracted text is long enough to analyze"""
    return len(text.split()) >= min_words