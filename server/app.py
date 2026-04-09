from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
import uvicorn

app = FastAPI(title="Medical Triage OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────────────────────────────
sessions = {}

# ── Patient scenarios ────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "id": "P1",
        "name": "James Wilson",
        "age": 72,
        "heart_rate": 118,
        "oxygen_saturation": 88.0,
        "systolic_bp": 85,
        "diastolic_bp": 55,
        "temperature": 37.2,
        "respiratory_rate": 24,
        "symptoms": ["chest pain", "shortness of breath"],
        "chief_complaint": "Severe chest pain radiating to left arm",
        "expected_action": "escalate",
        "urgency_score": 0.85,
        "status": "critical",
    },
    {
        "id": "P2",
        "name": "Sarah Johnson",
        "age": 34,
        "heart_rate": 72,
        "oxygen_saturation": 99.0,
        "systolic_bp": 118,
        "diastolic_bp": 78,
        "temperature": 36.8,
        "respiratory_rate": 16,
        "symptoms": ["mild headache"],
        "chief_complaint": "Mild headache for 2 hours",
        "expected_action": "discharge",
        "urgency_score": 0.15,
        "status": "stable",
    },
    {
        "id": "P3",
        "name": "Robert Chen",
        "age": 55,
        "heart_rate": 96,
        "oxygen_saturation": 93.0,
        "systolic_bp": 105,
        "diastolic_bp": 68,
        "temperature": 38.5,
        "respiratory_rate": 20,
        "symptoms": ["fever", "cough", "mild chest tightness"],
        "chief_complaint": "Fever and productive cough for 3 days",
        "expected_action": "treat",
        "urgency_score": 0.55,
        "status": "moderate",
    },
]

MAX_STEPS = len(SCENARIOS)


# ── Helper: NEWS score ────────────────────────────────────────────────────────
def calculate_news(p: dict) -> int:
    score = 0
    hr = p["heart_rate"]
    o2 = p["oxygen_saturation"]
    sbp = p["systolic_bp"]
    temp = p["temperature"]
    rr = p["respiratory_rate"]

    score += 3 if hr <= 40 or hr >= 131 else 2 if hr >= 111 else 1 if hr >= 91 or hr <= 50 else 0
    score += 3 if o2 <= 91 else 2 if o2 <= 93 else 1 if o2 <= 95 else 0
    score += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0
    score += 2 if temp <= 35.0 or temp >= 39.1 else 1 if temp <= 36.0 or temp >= 38.1 else 0
    score += 3 if rr <= 8 or rr >= 25 else 2 if rr >= 21 else 1 if rr >= 9 else 0
    return score


def news_to_risk(score: int) -> str:
    if score >= 6:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    return "LOW"


def make_observation(session: dict) -> dict:
    step = session["current_step"]
    done = step >= MAX_STEPS
    patients = []
    if not done:
        p = SCENARIOS[step]
        news = calculate_news(p)
        patients.append({
            "id": p["id"],
            "name": p["name"],
            "age": p["age"],
            "symptoms": p["symptoms"],
            "chief_complaint": p["chief_complaint"],
            "status": p["status"],
            "urgency_score": p["urgency_score"],
            "news_score": news,
            "risk_level": news_to_risk(news),
            "vitals": {
                "heart_rate": p["heart_rate"],
                "oxygen_saturation": p["oxygen_saturation"],
                "systolic_bp": p["systolic_bp"],
                "diastolic_bp": p["diastolic_bp"],
                "temperature": p["temperature"],
                "respiratory_rate": p["respiratory_rate"],
            },
        })
    return {
        "session_id": session["session_id"],
        "patients": patients,
        "current_step": step,
        "max_steps": MAX_STEPS,
        "done": done,
        "total_reward": session["total_reward"],
    }


# ── Pydantic models ──────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    action: str          # discharge | treat | escalate | investigate
    patient_id: str
    notes: Optional[str] = ""


class PredictRequest(BaseModel):
    age: int
    heart_rate: int
    oxygen_saturation: float
    systolic_bp: int
    diastolic_bp: int
    temperature: float
    symptoms: List[str] = []
    chief_complaint: str = ""


# ── OpenEnv required endpoints ───────────────────────────────────────────────

@app.post("/reset")
def reset(session_id: Optional[str] = None):
    """OpenEnv reset — creates/resets a session and returns initial observation."""
    sid = session_id or str(uuid.uuid4())
    sessions[sid] = {
        "session_id": sid,
        "current_step": 0,
        "total_reward": 0.0,
        "history": [],
    }
    return make_observation(sessions[sid])


@app.post("/step")
def step(body: StepRequest, session_id: Optional[str] = None):
    """OpenEnv step — apply an action and return next observation + reward."""
    sid = session_id or list(sessions.keys())[-1] if sessions else None
    if not sid or sid not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id. Call /reset first.")

    session = sessions[sid]
    step_idx = session["current_step"]

    if step_idx >= MAX_STEPS:
        raise HTTPException(status_code=400, detail="Episode already done. Call /reset to start again.")

    patient = SCENARIOS[step_idx]
    action = body.action.lower().strip()
    expected = patient["expected_action"]

    # Reward logic
    if action == expected:
        reward = 10.0
    elif action == "investigate":
        reward = 5.0
    elif expected == "escalate" and action == "discharge":
        reward = -8.0   # dangerous under-treatment
    elif expected == "discharge" and action == "escalate":
        reward = -5.0   # over-escalation
    else:
        reward = -5.0

    session["total_reward"] += reward
    session["history"].append({
        "step": step_idx + 1,
        "patient_id": body.patient_id,
        "action": action,
        "expected": expected,
        "reward": reward,
        "notes": body.notes,
    })
    session["current_step"] += 1

    obs = make_observation(session)
    obs["reward"] = reward
    obs["action_taken"] = action
    obs["expected_action"] = expected
    obs["correct"] = action == expected
    return obs


@app.get("/state")
def state(session_id: Optional[str] = None):
    """Return current state without advancing the episode."""
    sid = session_id or (list(sessions.keys())[-1] if sessions else None)
    if not sid or sid not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id.")
    return make_observation(sessions[sid])


# ── Extra utility endpoints ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Medical Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/stats", "/predict"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.get("/stats")
def stats():
    completed = [s for s in sessions.values() if s["current_step"] >= MAX_STEPS]
    return {
        "total_sessions": len(sessions),
        "completed_episodes": len(completed),
        "avg_reward": (
            sum(s["total_reward"] for s in completed) / len(completed)
            if completed else 0
        ),
        "max_steps_per_episode": MAX_STEPS,
    }


@app.post("/predict")
def predict(body: PredictRequest):
    """Standalone prediction — no session required."""
    p = {
        "heart_rate": body.heart_rate,
        "oxygen_saturation": body.oxygen_saturation,
        "systolic_bp": body.systolic_bp,
        "diastolic_bp": body.diastolic_bp,
        "temperature": body.temperature,
        "respiratory_rate": 18,
    }
    news = calculate_news(p)
    risk = news_to_risk(news)
    action_map = {"LOW": "discharge", "MEDIUM": "treat", "HIGH": "escalate"}
    action = action_map[risk]
    confidence = 0.95 if news >= 6 or news <= 1 else 0.78
    return {
        "news_score": news,
        "risk_level": risk,
        "action": action.upper(),
        "confidence": confidence,
        "reasoning": f"NEWS={news} → {risk} risk → {action.upper()}",
    }


# ── Entry point ───────────────────────────────────────────────
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()