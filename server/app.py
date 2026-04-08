"""
FastAPI Server for Medical Triage Environment - OpenEnv Compliant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Any
import uuid
import sys
import os
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the environment file
from medical_triage_env_environment import (
    MedicalTriageEnvironment, 
    MedicalAction
)

# ============================================================================
# Request/Response Models
# ============================================================================

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]

class OpenEnvResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Medical Triage Environment",
    description="OpenEnv compliant medical triage simulation for AI agent evaluation",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active environments
environments: Dict[str, MedicalTriageEnvironment] = {}
active_session: Optional[str] = None


# ============================================================================
# OpenEnv Required Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "medical-triage-env", "version": "2.0.0"}


@app.get("/metadata")
async def get_metadata():
    """OpenEnv metadata endpoint - Returns environment metadata"""
    return {
        "name": "medical-triage-env",
        "version": "2.0.0",
        "description": "Emergency Department Triage Environment for AI Agent Evaluation",
        "author": "Medical Triage Team",
        "license": "MIT",
        "tags": ["healthcare", "triage", "rl", "emergency-medicine"]
    }


@app.get("/schema")
async def get_schema():
    """OpenEnv schema endpoint - Returns action, observation, and state schemas"""
    return {
        "action": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["discharge", "treat", "escalate", "investigate"],
                    "description": "Type of medical action to perform"
                },
                "patient_id": {
                    "type": "string",
                    "description": "ID of the patient to act on"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional clinical notes"
                }
            },
            "required": ["type", "patient_id"]
        },
        "observation": {
            "type": "object",
            "properties": {
                "patients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "symptoms": {"type": "array", "items": {"type": "string"}},
                            "vitals": {
                                "type": "object",
                                "properties": {
                                    "heart_rate": {"type": "integer"},
                                    "blood_pressure_systolic": {"type": "integer"},
                                    "oxygen_saturation": {"type": "number"},
                                    "temperature": {"type": "number"}
                                }
                            },
                            "urgency_score": {"type": "number"},
                            "status": {"type": "string"}
                        }
                    }
                },
                "current_step": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "task_id": {"type": "string"},
                "done": {"type": "boolean"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "step_count": {"type": "integer"},
                "task_id": {"type": "string"},
                "resource_usage": {"type": "object"}
            }
        }
    }


@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """OpenEnv MCP (Model Context Protocol) endpoint"""
    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 1),
        "result": {
            "status": "ready",
            "capabilities": ["reset", "step", "observe"],
            "environment": "medical-triage-env",
            "version": "2.0.0"
        }
    }


@app.post("/reset")
async def reset_environment(request: ResetRequest) -> OpenEnvResponse:
    """OpenEnv reset endpoint"""
    global active_session
    
    # Create new session
    session_id = str(uuid.uuid4())[:8]
    env = MedicalTriageEnvironment()
    
    # Reset environment
    observation = env.reset(task_id=request.task_id, seed=request.seed)
    
    # Store
    environments[session_id] = env
    active_session = session_id
    
    # Convert to dict
    observation_dict = {
        "patients": [p.model_dump() for p in observation.patients],
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
    """OpenEnv step endpoint"""
    global active_session
    
    if active_session is None or active_session not in environments:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    
    env = environments[active_session]
    
    # Create action from request
    action = MedicalAction(
        type=request.action.get("type", ""),
        patient_id=request.action.get("patient_id", ""),
        notes=request.action.get("notes", "")
    )
    
    # Execute step
    observation, reward, done, info = env.step(action)
    
    # Convert to dict
    observation_dict = {
        "patients": [p.model_dump() for p in observation.patients],
        "current_step": observation.current_step,
        "max_steps": observation.max_steps,
        "task_id": observation.task_id,
        "done": observation.done
    }
    
    # Cleanup if done
    if done:
        del environments[active_session]
        active_session = None
    
    return OpenEnvResponse(
        observation=observation_dict,
        reward=reward,
        done=done,
        info=info
    )


@app.get("/state")
async def get_state():
    """Get current environment state"""
    if active_session is None or active_session not in environments:
        raise HTTPException(status_code=404, detail="No active session")
    
    env = environments[active_session]
    state = env.state()
    
    return {
        "step_count": state.step_count,
        "task_id": state.task_id,
        "resource_usage": state.resource_usage,
        "actions_taken": [a.dict() for a in state.actions_taken]
    }


@app.get("/")
async def root():
    return {
        "message": "Medical Triage Environment API",
        "version": "2.0.0",
        "openenv_compliant": True,
        "endpoints": {
            "health": "GET /health",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "mcp": "POST /mcp",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state"
        }
    }


# ============================================================================
# Fix the missing main() function
# ============================================================================

def main():
    """Main entry point for the environment server"""
    import uvicorn
    print("=" * 60)
    print("🏥 Medical Triage Environment Server - OpenEnv Compliant")
    print("=" * 60)
    print("Server running at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("Metadata: http://localhost:8000/metadata")
    print("Schema: http://localhost:8000/schema")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()