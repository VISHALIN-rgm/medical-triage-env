"""
FastAPI Server for Medical Triage Environment
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, List
import sys
import os
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MedicalAction
from server.medical_triage_env_environment import MedicalTriageEnvironment

# ============== Request/Response Models ==============

class PredictRequest(BaseModel):
    """Request model for /predict endpoint"""
    age: int
    heart_rate: int
    oxygen_saturation: float
    systolic_bp: int
    diastolic_bp: int
    temperature: float
    symptoms: List[str]
    chief_complaint: str = "Not specified"


class PredictResponse(BaseModel):
    """Response model for /predict endpoint"""
    news_score: int
    risk_level: str
    action: str
    confidence: float
    reasoning: str
    q_value: float


# ============== FastAPI App ==============

app = FastAPI(
    title="Medical Triage Environment",
    description="OpenEnv compliant medical triage simulation for AI agent evaluation",
    version="2.0.0"
)

environments: Dict[str, MedicalTriageEnvironment] = {}


# ============== Helper Functions ==============

def calculate_news_score(heart_rate: int, oxygen_saturation: float, temperature: float, systolic_bp: int) -> int:
    """Calculate NEWS score based on vital signs"""
    score = 0
    
    # Heart Rate
    hr = heart_rate
    if hr <= 40:
        score += 3
    elif hr <= 50:
        score += 1
    elif hr <= 90:
        score += 0
    elif hr <= 110:
        score += 1
    elif hr <= 130:
        score += 2
    else:
        score += 3
    
    # Oxygen Saturation
    o2 = oxygen_saturation
    if o2 <= 91:
        score += 3
    elif o2 <= 93:
        score += 2
    elif o2 <= 95:
        score += 1
    
    # Temperature
    temp = temperature
    if temp <= 35.0:
        score += 3
    elif temp <= 36.0:
        score += 1
    elif temp <= 38.0:
        score += 0
    elif temp <= 39.0:
        score += 1
    else:
        score += 2
    
    # Blood Pressure
    sbp = systolic_bp
    if sbp <= 90:
        score += 3
    elif sbp <= 100:
        score += 2
    elif sbp <= 110:
        score += 1
    elif sbp <= 219:
        score += 0
    else:
        score += 2
    
    return score


# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "message": "Medical Triage Environment API",
        "version": "2.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/predict"],
        "docs": "/docs"
    }


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


@app.delete("/reset/{session_id}")
async def delete_session(session_id: str):
    """Clean up environment session"""
    if session_id in environments:
        del environments[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Clinical prediction endpoint - Get triage recommendation based on patient vitals
    """
    try:
        # Calculate NEWS score
        news_score = calculate_news_score(
            heart_rate=request.heart_rate,
            oxygen_saturation=request.oxygen_saturation,
            temperature=request.temperature,
            systolic_bp=request.systolic_bp
        )
        
        # Determine risk level and action based on NEWS score
        if news_score >= 6:
            risk_level = "HIGH"
            action = "ESCALATE"
            confidence = 0.92
            reasoning = f"CRITICAL: NEWS={news_score}. HR={request.heart_rate}, O2={request.oxygen_saturation}%, BP={request.systolic_bp}. Immediate ICU escalation required."
        elif news_score >= 3:
            risk_level = "MEDIUM"
            action = "TREAT"
            confidence = 0.85
            reasoning = f"MODERATE: NEWS={news_score}. HR={request.heart_rate}, Temp={request.temperature:.1f}°C. Medical treatment required."
        else:
            risk_level = "LOW"
            action = "DISCHARGE"
            confidence = 0.90
            reasoning = f"LOW RISK: NEWS={news_score}. HR={request.heart_rate}, BP={request.systolic_bp}. Stable vitals, routine discharge."
        
        # Add symptom information to reasoning
        if request.symptoms:
            reasoning += f" Symptoms: {', '.join(request.symptoms[:2])}."
        
        return PredictResponse(
            news_score=news_score,
            risk_level=risk_level,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            q_value=8.5
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)