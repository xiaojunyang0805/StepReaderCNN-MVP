"""
Training API Backend
FastAPI backend for training operations (prepared for future async training).
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
import uuid
import json
from datetime import datetime

# Create FastAPI app
app = FastAPI(title="StepReaderCNN Training API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class TrainingConfig(BaseModel):
    """Training configuration."""
    model_name: str
    base_filters: int = 32
    dropout: float = 0.5
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 0.001
    early_stopping: bool = True
    patience: Optional[int] = 10
    use_class_weights: bool = True
    target_length: int = 10000
    normalize_method: str = 'zscore'
    augment_train: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class TrainingStatus(BaseModel):
    """Training status response."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class TrainingResult(BaseModel):
    """Training result response."""
    job_id: str
    status: str
    history: Optional[Dict[str, List[float]]] = None
    final_metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None


# Global training state (in production, use Redis or database)
training_jobs: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "StepReaderCNN Training API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/training/start", response_model=TrainingStatus)
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Start a new training job.

    Args:
        config: Training configuration
        background_tasks: FastAPI background tasks

    Returns:
        Training status with job ID
    """
    # Generate job ID
    job_id = str(uuid.uuid4())

    # Initialize job state
    training_jobs[job_id] = {
        "status": "pending",
        "config": config.dict(),
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "current_epoch": None,
        "total_epochs": config.num_epochs,
        "history": None,
        "error": None
    }

    # Add training task to background
    # background_tasks.add_task(run_training, job_id, config)

    return TrainingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        current_epoch=None,
        total_epochs=config.num_epochs,
        message="Training job queued"
    )


@app.get("/training/{job_id}/status", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """
    Get status of a training job.

    Args:
        job_id: Training job ID

    Returns:
        Current training status
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]

    return TrainingStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_epoch=job["current_epoch"],
        total_epochs=job["total_epochs"],
        metrics=job.get("metrics"),
        message=job.get("error")
    )


@app.get("/training/{job_id}/result", response_model=TrainingResult)
async def get_training_result(job_id: str):
    """
    Get result of a completed training job.

    Args:
        job_id: Training job ID

    Returns:
        Training result with history and metrics
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]

    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job is {job['status']}, not completed"
        )

    return TrainingResult(
        job_id=job_id,
        status=job["status"],
        history=job.get("history"),
        final_metrics=job.get("final_metrics"),
        model_path=job.get("model_path")
    )


@app.delete("/training/{job_id}")
async def cancel_training(job_id: str):
    """
    Cancel a running training job.

    Args:
        job_id: Training job ID

    Returns:
        Cancellation status
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]

    if job["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel {job['status']} job"
        )

    # Mark as cancelled
    job["status"] = "cancelled"

    return {"job_id": job_id, "status": "cancelled"}


@app.get("/training/jobs")
async def list_training_jobs():
    """
    List all training jobs.

    Returns:
        List of all training jobs with basic info
    """
    jobs = []
    for job_id, job in training_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "config": {
                "model_name": job["config"]["model_name"],
                "num_epochs": job["config"]["num_epochs"]
            }
        })

    return {"jobs": jobs, "total": len(jobs)}


@app.get("/models")
async def list_models():
    """
    List available trained models.

    Returns:
        List of saved models
    """
    models_dir = Path("outputs/models")

    if not models_dir.exists():
        return {"models": [], "total": 0}

    models = []
    for model_path in models_dir.glob("*.pth"):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            models.append({
                "name": model_path.stem,
                "path": str(model_path),
                "model_class": checkpoint.get("model_class", "Unknown"),
                "size_mb": model_path.stat().st_size / (1024 * 1024)
            })
        except Exception as e:
            print(f"Error loading {model_path}: {e}")

    return {"models": models, "total": len(models)}


@app.get("/datasets")
async def list_datasets():
    """
    List available datasets.

    Returns:
        List of available datasets
    """
    # This is a placeholder - implement based on your data structure
    return {
        "datasets": [
            {"name": "TestData", "path": "TestData", "samples": 42}
        ],
        "total": 1
    }


# Helper function for background training
async def run_training(job_id: str, config: TrainingConfig):
    """
    Run training in background (placeholder for future async implementation).

    Args:
        job_id: Training job ID
        config: Training configuration
    """
    try:
        # Update status
        training_jobs[job_id]["status"] = "running"

        # TODO: Implement actual training logic here
        # This would involve:
        # 1. Load data
        # 2. Create model
        # 3. Train with callbacks to update progress
        # 4. Save results

        # Placeholder
        import time
        for epoch in range(config.num_epochs):
            time.sleep(0.1)  # Simulate training
            training_jobs[job_id]["current_epoch"] = epoch + 1
            training_jobs[job_id]["progress"] = (epoch + 1) / config.num_epochs

        # Mark as completed
        training_jobs[job_id]["status"] = "completed"

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn

    print("Starting Training API server...")
    print("API documentation available at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
