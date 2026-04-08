"""
FastAPI Server for Medical Triage Environment
"""

from fastapi import FastAPI, HTTPException
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MedicalAction
from server.medical_triage_env_environment import MedicalTriageEnvironment

app = FastAPI(
    title="Medical Triage Environment",
    description="OpenEnv compliant medical triage simulation for AI agent evaluation",
    version="2.0.0"
)

environments: Dict[str, MedicalTriageEnvironment] = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medical-triage-env", "version": "2.0.0"}

@app.post("/reset")
async def reset_environment(session_id: str, task_id: str = "easy", num_patients: int = 3):
    """Reset environment with new patients"""
    env = MedicalTriageEnvironment()
    observation = env.reset(task_id=task_id, num_patients=num_patients)
    environments[session_id] = env
    return observation

@app.post("/step")
async def step_environment(session_id: str, action: MedicalAction):
    """Execute an action"""
    if session_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[session_id]
    observation, reward, done, info = env.step(action)
    
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
async def get_state(session_id: str):
    """Get current environment state"""
    if session_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    return environments[session_id].state()

@app.get("/")
async def root():
    return {
        "message": "Medical Triage Environment API",
        "endpoints": ["/health", "/reset", "/step", "/state"],
        "docs": "/docs"
    }


@app.delete("/reset/{session_id}")
async def delete_session(session_id: str):
    """Clean up environment session"""
    if session_id in environments:
        del environments[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)