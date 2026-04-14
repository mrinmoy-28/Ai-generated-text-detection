# backend/main.py
import sys
import io
sys.path.append("src")
sys.path.append("backend")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from phase6_ensemble import HybridDetector
from phase6_explainability import ExplainableDetector
from database import save_detection, get_history, get_stats
from file_parser import extract_text, validate_text
from batch import process_batch, extract_zip
from report import generate_report

# ── App Setup ─────────────────────────────────────────────
app      = FastAPI(title="AI Text Detector API", version="1.0")
detector = HybridDetector()
explainer = ExplainableDetector()

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request Models ─────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class ReportInput(BaseModel):
    text:       str
    verdict:    str
    confidence: float
    breakdown:  dict

# ── Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "running", "model": "loaded"}


@app.post("/detect")
async def detect_text(input: TextInput):
    if len(input.text.strip().split()) < 20:
        raise HTTPException(400, "Text too short — need at least 20 words")
    result = detector.detect(input.text)
    save_detection(input.text, result, source="text")
    return result


@app.post("/detect/sentences")
async def detect_sentences(input: TextInput):
    if len(input.text.strip().split()) < 20:
        raise HTTPException(400, "Text too short")
    sentences = detector.detect_sentences(input.text)
    return {"sentences": sentences}


@app.post("/detect/file")
async def detect_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        text = extract_text(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {str(e)}")

    if not validate_text(text):
        raise HTTPException(400, "File text too short to analyze")

    result             = detector.detect(text)
    result['filename'] = file.filename
    save_detection(text, result, source="file")
    return result


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 files per batch")

    file_list = []
    for f in files:
        # Handle ZIP files
        if f.filename.endswith('.zip'):
            zip_bytes  = await f.read()
            extracted  = extract_zip(zip_bytes)
            file_list.extend(extracted)
        else:
            file_bytes = await f.read()
            file_list.append((f.filename, file_bytes))

    results = process_batch(file_list, detector)

    # Save each to database
    for r in results['results']:
        save_detection(r.get('text', ''), r, source="batch")

    return results


@app.post("/explain")
async def explain_text(input: TextInput):
    if len(input.text.strip().split()) < 20:
        raise HTTPException(400, "Text too short")
    try:
        word_scores = explainer.explain(input.text)
        return {
            "top_ai_words":    [w for w in word_scores if w['push'] == 'AI'][:5],
            "top_human_words": [w for w in word_scores if w['push'] == 'Human'][:5]
        }
    except Exception as e:
        raise HTTPException(500, f"Explainability failed: {str(e)}")


@app.post("/report")
async def generate_pdf_report(input: ReportInput):
    result = {
        "verdict":    input.verdict,
        "confidence": input.confidence,
        "breakdown":  input.breakdown
    }

    # Get sentence analysis too
    sentences = None
    if len(input.text.split()) >= 20:
        sent_result = detector.detect_sentences(input.text)
        sentences   = sent_result

    pdf_bytes = generate_report(input.text, result, sentences)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=detection_report.pdf"}
    )


@app.get("/history")
async def history(limit: int = 50):
    return {"history": get_history(limit)}


@app.get("/stats")
async def stats():
    return get_stats()


# Run: uvicorn backend.main:app --reload --port 8000



## 🔁 How All Files Connect

#main.py
#  ├── imports HybridDetector      ← from src/phase6_ensemble.py
#  ├── imports ExplainableDetector ← from src/phase6_explainability.py
#  ├── imports database.py         ← save/fetch SQLite data
#  ├── imports file_parser.py      ← extract text from files
#  ├── imports batch.py            ← parallel multi-file processing
#  └── imports report.py           ← generate PDF