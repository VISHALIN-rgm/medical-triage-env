import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Medical Triage Environment")

# Store sessions
sessions: Dict[str, Any] = {}

# Models
class ResetRequest(BaseModel):
    task_id: str = "easy"

class ActionPayload(BaseModel):
    type: str
    patient_id: str
    notes: Optional[str] = ""

class StepRequest(BaseModel):
    action: ActionPayload
    session_id: Optional[str] = None

# Simple patient data
sample_patients = {
    "easy": [
        {"id": "P1", "name": "John Doe", "age": 65, "symptoms": ["chest pain"], "urgency_score": 0.8},
        {"id": "P2", "name": "Jane Smith", "age": 45, "symptoms": ["headache"], "urgency_score": 0.2},
        {"id": "P3", "name": "Bob Johnson", "age": 70, "symptoms": ["shortness of breath"], "urgency_score": 0.9},
    ],
    "medium": [
        {"id": "P1", "name": "Patient 1", "age": 50, "symptoms": ["fever"], "urgency_score": 0.5},
        {"id": "P2", "name": "Patient 2", "age": 60, "symptoms": ["cough"], "urgency_score": 0.6},
        {"id": "P3", "name": "Patient 3", "age": 40, "symptoms": ["fatigue"], "urgency_score": 0.4},
        {"id": "P4", "name": "Patient 4", "age": 55, "symptoms": ["nausea"], "urgency_score": 0.3},
        {"id": "P5", "name": "Patient 5", "age": 45, "symptoms": ["dizziness"], "urgency_score": 0.4},
    ],
    "hard": [
        {"id": "P1", "name": "Critical Patient", "age": 75, "symptoms": ["severe chest pain"], "urgency_score": 0.95},
        {"id": "P2", "name": "Deteriorating Patient", "age": 68, "symptoms": ["respiratory distress"], "urgency_score": 0.85},
        {"id": "P3", "name": "Unstable Patient", "age": 80, "symptoms": ["hypotension"], "urgency_score": 0.9},
    ],
}

@app.get("/")
async def root():
    return {"name": "medical-triage-env", "version": "1.0.0", "endpoints": ["/reset", "/step", "/health", "/stats"]}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# CRITICAL FIX: session_id as REQUIRED query parameter
@app.post("/reset")
async def reset(
    session_id: str = Query(..., description="Session ID (required)"),
    request: ResetRequest = None
):
    """Reset endpoint - session_id is REQUIRED as query parameter"""
    
    task_id = request.task_id if request else "easy"
    
    if task_id not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task_id")
    
    # Create session
    sessions[session_id] = {
        "task_id": task_id,
        "step": 0,
        "done": False,
        "patients": sample_patients[task_id].copy(),
        "current_patient_idx": 0
    }
    
    return {
        "task_id": task_id,
        "session_id": session_id,
        "current_step": 0,
        "max_steps": {"easy": 10, "medium": 20, "hard": 25}[task_id],
        "done": False,
        "patients": sessions[session_id]["patients"],
        "message": f"Episode started for task={task_id}"
    }

@app.post("/step")
async def step(request: StepRequest):
    """Step endpoint"""
    
    session_id = request.session_id
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found. Call /reset first")
    
    sess = sessions[session_id]
    
    if sess["done"]:
        raise HTTPException(status_code=400, detail="Episode already finished")
    
    # Process action
    action = request.action.type
    patient_id = request.action.patient_id
    
    # Simple reward calculation
    reward = 10.0 if action in ["treat", "escalate"] else 5.0
    
    # Move to next patient
    sess["current_patient_idx"] += 1
    sess["step"] += 1
    sess["done"] = sess["current_patient_idx"] >= len(sess["patients"])
    
    return {
        "observation": {
            "task_id": sess["task_id"],
            "current_step": sess["step"],
            "max_steps": {"easy": 10, "medium": 20, "hard": 25}[sess["task_id"]],
            "done": sess["done"],
            "patients": sess["patients"][sess["current_patient_idx"]:] if not sess["done"] else []
        },
        "reward": reward,
        "done": sess["done"],
        "info": {}
    }

@app.get("/stats")
async def get_stats():
    return {"total_sessions": len(sessions), "status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)