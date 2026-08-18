"""
FastAPI REST API Backend for Real-Time Multilingual Translation System.
Serves universal translation endpoints, dynamic language registry, media upload processing queue, and downloads.
Directly redirects root `/` to the interactive Web Dashboard.
"""

import os
import uuid
import threading
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from src.language.registry import get_supported_languages, resolve_language_code, get_language_display_name
from src.translation.engine import translation_engine
from src.pipeline import pipeline
from src.storage.db import get_translation_history
from src.translation.model_registry import model_registry

app = FastAPI(
    title="Real-Time Multilingual Translation API",
    description="Universal Multilingual Audio-Visual Translation, Subtitle & Audio Dubbing System (Local Inference, Zero API Keys)",
    version="1.0.0"
)

# Enable CORS for local web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory Async Job Store
JOBS_STORE: Dict[str, Dict[str, Any]] = {}
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


class TextTranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "tel_Telu"
    target_langs: Optional[List[str]] = None


@app.get("/")
def read_root():
    """Redirects directly to the Web Dashboard index page when server starts."""
    return RedirectResponse(url="/dashboard/index.html")


@app.get("/api/info")
def get_info():
    return {
        "system": "Real-Time Multilingual Translation & Audio Dubbing Subtitle Generator",
        "status": "Online",
        "device": model_registry.device,
        "api_keys_required": False
    }


@app.get("/api/languages")
def get_languages():
    """Returns dynamic registry of supported languages for frontend language selectors."""
    return {
        "total_count": len(get_supported_languages()),
        "languages": get_supported_languages()
    }


@app.get("/api/status")
def get_system_status():
    """Returns hardware & model status."""
    return {
        "engine_ready": True,
        "device": model_registry.device,
        "primary_model": model_registry.primary_model_name,
        "execution": "100% Local Inference",
        "api_key_required": False,
        "languages_count": len(get_supported_languages())
    }


@app.post("/api/translate/text")
def translate_text(req: TextTranslationRequest):
    """Processes universal text translation for single or multiple target languages."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if req.target_langs and len(req.target_langs) > 1:
        res = translation_engine.translate_multi_target(req.text, req.source_lang, req.target_langs)
        return {"status": "success", "result": res}
    else:
        res = translation_engine.translate_single(req.text, req.source_lang, req.target_lang)
        return {"status": "success", "result": res}


@app.post("/api/process/media")
async def process_media_file(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("tel_Telu")
):
    """Uploads media file and launches asynchronous audio/video processing + dubbing pipeline job."""
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    saved_filename = f"{job_id}{file_ext}"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)

    with open(saved_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    JOBS_STORE[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "stage": "Job queued",
        "progress_pct": 5,
        "file_name": file.filename,
        "result": None
    }

    # Worker Thread Callback
    def run_pipeline_task():
        def progress_cb(stage: str, pct: int):
            JOBS_STORE[job_id]["stage"] = stage
            JOBS_STORE[job_id]["progress_pct"] = pct

        try:
            res = pipeline.process_media(
                file_path=saved_path,
                source_lang=source_lang,
                target_langs=target_lang,
                progress_callback=progress_cb
            )
            JOBS_STORE[job_id]["status"] = "completed"
            JOBS_STORE[job_id]["result"] = res
        except Exception as e:
            JOBS_STORE[job_id]["status"] = "failed"
            JOBS_STORE[job_id]["error"] = str(e)

    thread = threading.Thread(target=run_pipeline_task)
    thread.start()

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Media upload successful. Processing started."
    }


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Fetches real-time status and stage progress of processing job."""
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return JOBS_STORE[job_id]


@app.get("/api/downloads/{file_name}")
def download_output_file(file_name: str):
    """Downloads generated SRT, VTT, CSV, or video overlay files."""
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    file_path = os.path.join(outputs_dir, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")

    return FileResponse(file_path, filename=file_name)


@app.get("/api/history")
def get_history():
    """Retrieves SQLite translation history."""
    return {
        "history": get_translation_history(50)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
