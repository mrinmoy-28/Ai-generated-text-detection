# backend/batch.py
import os
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from file_parser import extract_text, validate_text

def process_batch(files: list, detector) -> dict:
    """
    Process multiple files in parallel.
    files: list of (filename, file_bytes) tuples
    detector: HybridDetector instance
    """
    results     = []
    failed      = []

    def process_single(filename, file_bytes):
        try:
            # Extract text
            text = extract_text(file_bytes, filename)

            # Validate
            if not validate_text(text):
                return {
                    "filename": filename,
                    "status":   "failed",
                    "error":    "Text too short to analyze"
                }

            # Detect
            result = detector.detect(text)
            return {
                "filename":   filename,
                "status":     "success",
                "verdict":    result['verdict'],
                "confidence": result['confidence'],
                "breakdown":  result['breakdown']
            }

        except Exception as e:
            return {
                "filename": filename,
                "status":   "failed",
                "error":    str(e)
            }

    # Run in parallel — max 4 workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single, fname, fbytes): fname
            for fname, fbytes in files
        }

        for future in as_completed(futures):
            result = future.result()
            if result['status'] == 'success':
                results.append(result)
            else:
                failed.append(result)

    # Summary
    ai_count    = sum(1 for r in results if r['verdict'] == 'AI Generated')
    human_count = sum(1 for r in results if r['verdict'] == 'Human Written')
    avg_conf    = (
        sum(r['confidence'] for r in results) / len(results)
        if results else 0
    )

    return {
        "results": results,
        "failed":  failed,
        "summary": {
            "total":          len(files),
            "processed":      len(results),
            "failed":         len(failed),
            "ai_count":       ai_count,
            "human_count":    human_count,
            "avg_confidence": round(avg_conf, 1)
        }
    }


def extract_zip(zip_bytes: bytes) -> list:
    """
    Extract all text files from a ZIP archive.
    Returns list of (filename, file_bytes) tuples.
    """
    files      = []
    supported  = {'.txt', '.pdf', '.docx'}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            ext = os.path.splitext(name)[1].lower()
            # Skip hidden files and unsupported types
            if ext in supported and not name.startswith('__'):
                file_bytes = zf.read(name)
                files.append((os.path.basename(name), file_bytes))

    return files