"""
Veloura Agentic AI - FastAPI Backend Server
Exposes clinical case processing, multi-agent orchestration, and real-time productivity metrics.
"""
import os
import io
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

from veloura_backend.agents.master_coordinator import MasterCoordinatorAgent
from veloura_backend.agents.productivity_tracker import ProductivityTracker
from veloura_backend.config import CLASS_INFO, CLASS_NAMES

app = FastAPI(
    title="Veloura Agentic AI - Clinical Dermatology Decision Support",
    version="2.0.0",
    description="Multi-Agent diagnostic pipeline with real-time productivity tracking."
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Master Coordinator and Productivity Tracker
coordinator = MasterCoordinatorAgent()
tracker = ProductivityTracker()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/")
def root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {
        "service": "Veloura Agentic AI Backend",
        "status": "ONLINE",
        "dashboard_url": "/dashboard/",
        "docs_url": "/docs"
    }


@app.get("/api/health")
def health():
    return {
        "status": "HEALTHY",
        "system": "Veloura Agentic AI 2026",
        "device": str(coordinator.diagnosis_agent.device),
        "architecture": coordinator.diagnosis_agent.architecture,
        "loaded_checkpoint": coordinator.diagnosis_agent.checkpoint_name,
        "classes_supported": CLASS_NAMES,
        "active_agents": [
            "ImageQualityAgent",
            "LesionAnalysisAgent",
            "DiagnosisAgent",
            "ReasoningContextAgent",
            "SafetyAgent",
            "MasterCoordinatorAgent"
        ]
    }


@app.post("/api/cases/analyze")
async def analyze_case(
    image: UploadFile = File(...),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    anatomical_site: Optional[str] = Form(None),
    evolution: Optional[str] = Form("STABLE"),
    history_of_melanoma: Optional[bool] = Form(False)
):
    try:
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    patient_context = {
        "age": age,
        "sex": sex,
        "anatomical_site": anatomical_site,
        "evolution": evolution,
        "history_of_melanoma": history_of_melanoma
    }

    # Run full multi-agent orchestration
    case_record = coordinator.process_case(
        image_pil=image_pil,
        patient_context=patient_context,
        case_id=f"CASE-{image.filename.split('.')[0][:12]}"
    )

    # Record in telemetry & productivity tracker
    tracker.record_case(case_record)

    return case_record


@app.get("/api/productivity/metrics")
def get_productivity_metrics():
    return tracker.get_metrics()


@app.post("/api/productivity/reset")
def reset_productivity_metrics():
    global tracker
    tracker = ProductivityTracker()
    return {"status": "RESET_SUCCESSFUL"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("veloura_backend.main:app", host="0.0.0.0", port=8000, reload=True)
