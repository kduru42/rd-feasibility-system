#!/usr/bin/env python3
"""
R&D Center Feasibility Evaluation System - Web Server

FastAPI server that wraps rd_center_evaluator.py and serves the frontend.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import our evaluator module
from rd_center_evaluator import (
    RD_CENTER_CRITERIA,
    extract_text_from_pdf,
    get_pdf_metadata,
    run_evaluation,
    generate_report,
)

# ============================================================================
# Configuration
# ============================================================================

UPLOAD_DIR = Path("uploads").resolve()
REPORT_DIR = Path("reports").resolve()
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-4-turbo"}
FILE_ID_PATTERN = re.compile(r"^[a-f0-9]{8}$")

app = FastAPI(
    title="R&D Center Feasibility Evaluation System",
    version="1.0.0",
    description="AI-powered evaluation of company documents against Turkish R&D center criteria (Law No. 5746)",
)

# CORS: allow same-origin by default, add your domain for production
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# In-memory store: file_id -> { file_path, filename, ... }
uploads: dict = {}
evaluations: dict = {}


# ============================================================================
# Helpers
# ============================================================================

def validate_file_id(file_id: str):
    """Validates file_id format to prevent injection."""
    if not FILE_ID_PATTERN.match(file_id):
        raise HTTPException(400, "Invalid file ID format.")


def sanitize_filename(filename: str) -> str:
    """Removes path separators and dangerous characters from filename."""
    name = os.path.basename(filename)
    name = re.sub(r'[^\w.\-]', '_', name)
    return name


def resolve_upload_path(file_id: str) -> Path:
    """Safely resolves upload path using server-side lookup only."""
    validate_file_id(file_id)
    if file_id not in uploads:
        raise HTTPException(404, "File not found. Please upload again.")
    path = Path(uploads[file_id]["file_path"]).resolve()
    # Path traversal protection: ensure file is inside UPLOAD_DIR
    if not str(path).startswith(str(UPLOAD_DIR)):
        raise HTTPException(403, "Access denied.")
    if not path.exists():
        raise HTTPException(404, "File not found on disk.")
    return path


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/criteria")
async def get_criteria():
    """Returns all evaluation criteria grouped by category."""
    return {
        "mandatory": {k: v for k, v in RD_CENTER_CRITERIA.items() if v["category"] == "mandatory"},
        "performance": {k: v for k, v in RD_CENTER_CRITERIA.items() if v["category"] == "performance"},
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF and extracts text + metadata."""
    # Validate extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # Read with size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB.")

    # Generate safe file path
    file_id = uuid.uuid4().hex[:8]
    safe_name = sanitize_filename(file.filename)
    file_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        text = extract_text_from_pdf(str(file_path))
        metadata = get_pdf_metadata(str(file_path))
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(422, f"PDF processing failed: {e}")

    # Store file reference server-side (never trust client for path)
    uploads[file_id] = {
        "file_path": str(file_path),
        "filename": file.filename,
        "metadata": metadata,
        "text_length": len(text),
    }

    return {
        "file_id": file_id,
        "filename": file.filename,
        "metadata": metadata,
        "text_length": len(text),
        "preview": text[:500] + "..." if len(text) > 500 else text,
    }


@app.post("/api/evaluate")
async def evaluate(
    file_id: str = Form(...),
    api_key: str = Form(...),
    model: str = Form("gpt-4o-mini"),
):
    """Runs LLM evaluation on an uploaded PDF."""
    # Validate file_id (server-side lookup, no client path)
    file_path = resolve_upload_path(file_id)

    # Validate model against whitelist
    if model not in ALLOWED_MODELS:
        raise HTTPException(400, f"Invalid model. Allowed: {', '.join(ALLOWED_MODELS)}")

    # Basic API key format check (not stored, just validated)
    if not api_key.startswith("sk-"):
        raise HTTPException(400, "Invalid API key format.")

    # Extract text
    try:
        pdf_text = extract_text_from_pdf(str(file_path))
        pdf_meta = get_pdf_metadata(str(file_path))
    except Exception as e:
        raise HTTPException(422, f"PDF read error: {e}")

    # Run evaluation via evaluator module
    try:
        result = run_evaluation(api_key, pdf_text, model)
    except Exception as e:
        raise HTTPException(500, f"LLM evaluation failed: {e}")

    # Generate and save report
    validate_file_id(file_id)
    report_md = generate_report(result, pdf_meta)
    report_path = REPORT_DIR / f"{file_id}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Save JSON
    json_path = REPORT_DIR / f"{file_id}_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Store in memory
    evaluations[file_id] = {
        "result": result,
        "report_path": str(report_path),
        "json_path": str(json_path),
        "pdf_metadata": pdf_meta,
    }

    return {
        "file_id": file_id,
        "evaluation": result,
    }


@app.get("/api/evaluations/{file_id}")
async def get_evaluation(file_id: str):
    """Retrieves a stored evaluation result."""
    validate_file_id(file_id)
    if file_id not in evaluations:
        raise HTTPException(404, "Evaluation not found.")
    return evaluations[file_id]


@app.get("/api/reports/{file_id}")
async def download_report(file_id: str):
    """Downloads the Markdown report."""
    validate_file_id(file_id)
    path = (REPORT_DIR / f"{file_id}_report.md").resolve()
    if not str(path).startswith(str(REPORT_DIR)) or not path.exists():
        raise HTTPException(404, "Report not found.")
    return FileResponse(str(path), media_type="text/markdown", filename=f"evaluation_{file_id}.md")


@app.get("/api/reports/{file_id}/json")
async def download_json(file_id: str):
    """Downloads the raw JSON result."""
    validate_file_id(file_id)
    path = (REPORT_DIR / f"{file_id}_result.json").resolve()
    if not str(path).startswith(str(REPORT_DIR)) or not path.exists():
        raise HTTPException(404, "JSON not found.")
    return FileResponse(str(path), media_type="application/json", filename=f"evaluation_{file_id}.json")


# ============================================================================
# Serve Frontend
# ============================================================================

@app.get("/")
async def serve_index():
    """Serves the frontend HTML file."""
    if Path("index.html").exists():
        return FileResponse("index.html")
    return {"message": "R&D Center Feasibility Evaluation API", "docs": "/docs"}