"""
FastAPI Server for Medical Triage Environment - OpenEnv Compliant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Any
import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel
from server.medical_triage_env_environment import MedicalTriageEnvironment

# OpenEnv Request/Response Models
class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]
    task_id: Optional[str] = None

class OpenEnvResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="Medical Triage Environment",
    description="OpenEnv compliant medical triage simulation for AI agent evaluation",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store environments
environments: Dict[str, MedicalTriageEnvironment] = {}
current_session: Optional[str] = None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medical-triage-env", "version": "2.0.0"}


@app.post("/reset")
async def reset_environment(request: ResetRequest) -> OpenEnvResponse:
    """OpenEnv compliant reset endpoint"""
    global current_session
    
    # Create new session
    session_id = str(uuid.uuid4())
    env = MedicalTriageEnvironment()
    
    observation = env.reset(task_id=request.task_id, seed=request.seed)
    
    environments[session_id] = env
    current_session = session_id
    
    # Convert observation to dict
    observation_dict = {
        "patients": [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in observation.patients],
        "current_step": observation.current_step,
        "max_steps": observation.max_steps,
        "task_id": observation.task_id,
        "done": observation.done
    }
    
    return OpenEnvResponse(
        observation=observation_dict,
        reward=0.0,
        done=False,
        info={"session_id": session_id, "message": f"Reset {request.task_id} task"}
    )


@app.post("/step")
async def step_environment(request: StepRequest) -> OpenEnvResponse:
    """OpenEnv compliant step endpoint"""
    global current_session
    
    if current_session is None or current_session not in environments:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    
    env = environments[current_session]
    
    # Convert action dict to MedicalAction
    from server.medical_triage_env_environment import MedicalAction
    action_data = request.action
    action = MedicalAction(
        type=action_data.get("type", ""),
        patient_id=action_data.get("patient_id", ""),
        notes=action_data.get("notes", "")
    )
    
    observation, reward, done, info = env.step(action)
    
    # Convert observation to dict
    observation_dict = {
        "patients": [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in observation.patients],
        "current_step": observation.current_step,
        "max_steps": observation.max_steps,
        "task_id": observation.task_id,
        "done": observation.done
    }
    
    # Clean up if done
    if done:
        del environments[current_session]
        current_session = None
    
    return OpenEnvResponse(
        observation=observation_dict,
        reward=reward,
        done=done,
        info=info
    )


@app.get("/")
async def root():
    return {
        "message": "Medical Triage Environment API - OpenEnv Compliant",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset",
            "step": "POST /step"
        }
    }


def main():
    import uvicorn
    print("=" * 60)
    print("🏥 Medical Triage Environment Server - OpenEnv Compliant")
    print("=" * 60)
    print("📍 Server running at: http://0.0.0.0:8001")
    print("📚 API Documentation: http://localhost:8001/docs")
    print("❤️  Health Check: http://localhost:8001/health")
    print("🔄 Reset: POST /reset with {\"task_id\": \"easy\"}")
    print("⚡ Step: POST /step with {\"action\": {\"type\": \"escalate\", \"patient_id\": \"P1\"}}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()