import os
import sys
import json
import time
import random
import pathlib
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from inference.py modules
from models import Patient, VitalSigns

# Try to import the real agent components
try:
    from inference import (
        RealClinicalAgent, RealDataLoader, PersistentQLearning,
        calculate_news_score, get_risk_level, get_guideline_action,
        ACTIONS, ACTION_ICON, RiskLevel, ClinicalDecision,
        INVESTIGATE_CONFIDENCE_THRESHOLD, ERROR_RATE, HF_TOKEN,
        OPENAI_AVAILABLE, DATASETS_AVAILABLE, MODEL_NAME, API_BASE_URL
    )
    INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from inference.py: {e}", file=sys.stderr)
    INFERENCE_AVAILABLE = False

import gradio as gr
import numpy as np

# =============================================================
# CONFIGURATION
# =============================================================
MODEL_VERSION = "32.0.0"
PORT = int(os.getenv("PORT", 7860))

# Global agent instances
_agent = None
_data_loader = None

# =============================================================
# SIMPLIFIED AGENT (if inference.py not available)
# =============================================================

class SimplifiedTriageAgent:
    """Fallback agent when inference.py is not available"""
    
    def __init__(self):
        self.actions = ["DISCHARGE", "TREAT", "ESCALATE", "INVESTIGATE"]
        self.q_table = {
            "LOW": {"DISCHARGE": 10.0, "TREAT": 1.0, "ESCALATE": 1.0, "INVESTIGATE": 3.0},
            "MEDIUM": {"DISCHARGE": 0.1, "TREAT": 8.5, "ESCALATE": 2.5, "INVESTIGATE": 3.5},
            "HIGH": {"DISCHARGE": -1.5, "TREAT": 1.0, "ESCALATE": 10.0, "INVESTIGATE": 3.0},
        }
    
    def assess_patient(self, patient_data: dict) -> dict:
        # Calculate NEWS score
        news_score = self._calculate_news(patient_data)
        
        # Determine risk level
        if news_score <= 1:
            risk = "LOW"
            guideline = "DISCHARGE"
        elif news_score <= 5:
            risk = "MEDIUM"
            guideline = "TREAT"
        else:
            risk = "HIGH"
            guideline = "ESCALATE"
        
        # Get Q-value
        q_value = self.q_table[risk].get(guideline, 5.0)
        
        # Calculate confidence
        confidence = min(0.95, 0.7 + (news_score / 20))
        
        # Generate reasoning
        reasoning = self._generate_reasoning(patient_data, news_score, risk, guideline)
        
        return {
            "news_score": news_score,
            "risk_level": risk,
            "action": guideline,
            "confidence": confidence,
            "reasoning": reasoning,
            "q_value": q_value
        }
    
    def _calculate_news(self, patient: dict) -> int:
        score = 0
        
        # Heart Rate
        hr = patient.get("heart_rate", 80)
        if hr <= 40 or hr >= 131:
            score += 3
        elif 41 <= hr <= 50 or 111 <= hr <= 130:
            score += 2
        elif 51 <= hr <= 55 or 106 <= hr <= 110:
            score += 1
        
        # Oxygen Saturation
        o2 = patient.get("oxygen_saturation", 96)
        if o2 <= 91:
            score += 3
        elif o2 <= 93:
            score += 2
        elif o2 <= 95:
            score += 1
        
        # Temperature
        temp = patient.get("temperature", 37.0)
        if temp <= 35.0 or temp >= 39.1:
            score += 2
        elif 38.1 <= temp <= 39.0:
            score += 1
        
        # Systolic BP
        sbp = patient.get("systolic_bp", 120)
        if sbp <= 90:
            score += 3
        elif sbp <= 100:
            score += 2
        
        # Respiratory Rate
        rr = patient.get("respiratory_rate", 16)
        if rr <= 8 or rr >= 25:
            score += 3
        elif 21 <= rr <= 24:
            score += 2
        elif 9 <= rr <= 11:
            score += 1
        
        return score
    
    def _generate_reasoning(self, patient: dict, news_score: int, risk: str, action: str) -> str:
        if risk == "HIGH":
            return f"Critical condition detected (NEWS={news_score}). Patient requires immediate {action} due to abnormal vital signs."
        elif risk == "MEDIUM":
            return f"Moderate clinical risk (NEWS={news_score}). Recommended {action} with monitoring."
        else:
            return f"Patient is stable (NEWS={news_score}). {action} with routine follow-up recommended."


def initialize_agent():
    """Initialize the triage agent"""
    global _agent, _data_loader
    
    if INFERENCE_AVAILABLE and not _agent:
        try:
            print("Loading real clinical agent from inference.py...", file=sys.stderr)
            _data_loader = RealDataLoader()
            _agent = RealClinicalAgent(_data_loader)
            print("✅ Real agent loaded successfully!", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Failed to load real agent: {e}", file=sys.stderr)
            print("Using simplified agent instead...", file=sys.stderr)
            _agent = SimplifiedTriageAgent()
    else:
        print("Using simplified triage agent...", file=sys.stderr)
        _agent = SimplifiedTriageAgent()
    
    return _agent


# =============================================================
# GRADIO INTERFACE FUNCTIONS
# =============================================================

def triage_patient(
    age: int,
    heart_rate: int,
    oxygen_saturation: float,
    systolic_bp: int,
    diastolic_bp: int,
    temperature: float,
    respiratory_rate: int,
    symptoms: str,
    chief_complaint: str
):
    """Main triage function for Gradio interface"""
    
    # Initialize agent if needed
    if _agent is None:
        initialize_agent()
    
    # Parse symptoms
    symptoms_list = [s.strip() for s in symptoms.split(",") if s.strip()]
    if not symptoms_list:
        symptoms_list = ["none reported"]
    
    # Prepare patient data
    patient_data = {
        "age": age,
        "heart_rate": heart_rate,
        "oxygen_saturation": oxygen_saturation,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "temperature": temperature,
        "respiratory_rate": respiratory_rate,
        "symptoms": symptoms_list,
        "chief_complaint": chief_complaint if chief_complaint else "Not specified"
    }
    
    # Get assessment
    try:
        if hasattr(_agent, 'assess_patient'):
            result = _agent.assess_patient(patient_data)
        else:
            # Create a temporary patient object for the real agent
            from models import VitalSigns, Patient
            temp_patient = Patient(
                id="temp",
                name="Patient",
                age=age,
                gender="Unknown",
                symptoms=symptoms_list,
                vitals=VitalSigns(
                    heart_rate=heart_rate,
                    blood_pressure_systolic=systolic_bp,
                    blood_pressure_diastolic=diastolic_bp,
                    oxygen_saturation=oxygen_saturation,
                    temperature=temperature,
                    respiratory_rate=respiratory_rate,
                ),
                status="stable",
                urgency_score=min(1.0, max(0.0, (heart_rate - 40) / 140.0)),
                time_to_deterioration=5,
                chief_complaint=chief_complaint,
                medical_history=[]
            )
            decision = _agent.make_decision(temp_patient)
            result = {
                "news_score": decision.news_score,
                "risk_level": decision.risk_level.value,
                "action": decision.action.upper(),
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "q_value": decision.q_value
            }
    except Exception as e:
        # Fallback to simple calculation
        news_score = _calculate_simple_news(heart_rate, oxygen_saturation, temperature, systolic_bp, respiratory_rate)
        if news_score <= 1:
            risk, action = "LOW", "DISCHARGE"
        elif news_score <= 5:
            risk, action = "MEDIUM", "TREAT"
        else:
            risk, action = "HIGH", "ESCALATE"
        
        result = {
            "news_score": news_score,
            "risk_level": risk,
            "action": action,
            "confidence": 0.85,
            "reasoning": f"NEWS score {news_score} indicates {risk} risk level. Recommended action: {action}.",
            "q_value": 5.0
        }
    
    # Create HTML output
    risk_colors = {
        "LOW": "#27ae60",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c"
    }
    
    action_icons = {
        "DISCHARGE": "✅",
        "TREAT": "💊",
        "ESCALATE": "🚨",
        "INVESTIGATE": "🔬"
    }
    
    risk_color = risk_colors.get(result["risk_level"], "#3498db")
    action_icon = action_icons.get(result["action"], "🏥")
    
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div style="background: white; border-radius: 10px; padding: 20px;">
            <h2 style="color: #2c3e50; margin-top: 0;">🏥 Triage Assessment Results</h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <strong>📊 NEWS Score:</strong><br>
                    <span style="font-size: 24px; font-weight: bold;">{result["news_score"]}</span>
                </div>
                
                <div style="background: {risk_color}20; padding: 15px; border-radius: 8px; border-left: 4px solid {risk_color};">
                    <strong>⚠️ Risk Level:</strong><br>
                    <span style="font-size: 20px; font-weight: bold; color: {risk_color};">{result["risk_level"]}</span>
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <strong>🎯 Recommended Action:</strong><br>
                    <span style="font-size: 20px; font-weight: bold;">{action_icon} {result["action"]}</span>
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <strong>📈 Confidence:</strong><br>
                    <span style="font-size: 20px; font-weight: bold;">{result["confidence"]:.1%}</span>
                </div>
            </div>
            
            <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <strong>💡 Clinical Reasoning:</strong><br>
                <p style="margin: 10px 0 0 0; line-height: 1.5;">{result["reasoning"]}</p>
            </div>
            
            <div style="background: #f0f0f0; padding: 10px; border-radius: 8px; font-size: 12px; color: #666;">
                <strong>🤖 Q-Value:</strong> {result["q_value"]:.2f} | 
                <strong>Model:</strong> {MODEL_VERSION}
            </div>
        </div>
    </div>
    """
    
    return html


def _calculate_simple_news(hr, o2, temp, sbp, rr):
    """Simple NEWS calculation fallback"""
    score = 0
    if hr <= 40 or hr >= 131: score += 3
    elif hr <= 50 or hr >= 111: score += 2
    elif hr <= 55 or hr >= 106: score += 1
    
    if o2 <= 91: score += 3
    elif o2 <= 93: score += 2
    elif o2 <= 95: score += 1
    
    if temp <= 35.0 or temp >= 39.1: score += 2
    elif temp >= 38.1: score += 1
    
    if sbp <= 90: score += 3
    elif sbp <= 100: score += 2
    
    if rr <= 8 or rr >= 25: score += 3
    elif rr >= 21: score += 2
    elif rr <= 11: score += 1
    
    return score


# =============================================================
# HEALTH CHECK ENDPOINT (for FastAPI compatibility)
# =============================================================

def health_check():
    """Health check function"""
    return {
        "status": "healthy",
        "version": MODEL_VERSION,
        "agent_ready": _agent is not None
    }


# =============================================================
# GRADIO INTERFACE
# =============================================================

# Create the Gradio interface
with gr.Blocks(title="🏥 Medical Triage AI Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Medical Triage AI Agent
    
    **AI-powered emergency department triage system** using NEWS (National Early Warning Score) and Q-learning
    
    Enter patient vital signs and symptoms to receive clinical triage recommendations.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Patient Information")
            age = gr.Slider(0, 120, label="Age (years)", value=65, step=1)
            chief_complaint = gr.Textbox(label="Chief Complaint", placeholder="e.g., Chest pain, Shortness of breath", lines=2)
            symptoms = gr.Textbox(label="Symptoms (comma-separated)", placeholder="chest pain, shortness of breath, fever", value="chest pain")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Vital Signs")
            heart_rate = gr.Slider(0, 200, label="Heart Rate (bpm)", value=88, step=1)
            oxygen_saturation = gr.Slider(50, 100, label="Oxygen Saturation (%)", value=95, step=1)
            systolic_bp = gr.Slider(50, 200, label="Systolic BP (mmHg)", value=120, step=1)
            diastolic_bp = gr.Slider(30, 150, label="Diastolic BP (mmHg)", value=80, step=1)
            temperature = gr.Slider(35.0, 41.0, label="Temperature (°C)", value=37.0, step=0.1)
            respiratory_rate = gr.Slider(5, 40, label="Respiratory Rate (breaths/min)", value=16, step=1)
    
    with gr.Row():
        triage_btn = gr.Button("🩺 Perform Triage", variant="primary", size="lg")
    
    with gr.Row():
        output = gr.HTML(label="Assessment Results")
    
    # Example cases
    gr.Markdown("### 📚 Example Cases")
    with gr.Row():
        gr.Examples(
            examples=[
                [45, 72, 98, 118, 78, 37.0, 16, "none", "Routine checkup"],
                [72, 120, 88, 85, 55, 37.2, 24, "chest pain, shortness of breath", "Severe chest pain radiating to left arm"],
                [38, 110, 92, 95, 60, 38.5, 22, "fever, cough, fatigue", "Fever and productive cough for 3 days"],
                [28, 130, 96, 110, 70, 36.8, 28, "wheezing, difficulty breathing", "Asthma exacerbation"],
            ],
            inputs=[age, heart_rate, oxygen_saturation, systolic_bp, diastolic_bp, temperature, respiratory_rate, symptoms, chief_complaint]
        )
    
    # Set up the trigger
    triage_btn.click(
        fn=triage_patient,
        inputs=[age, heart_rate, oxygen_saturation, systolic_bp, diastolic_bp, temperature, respiratory_rate, symptoms, chief_complaint],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### 📖 About
    
    - **NEWS Score**: National Early Warning Score (0-20+)
    - **Risk Levels**: LOW (0-1), MEDIUM (2-5), HIGH (6+)
    - **Actions**: DISCHARGE, TREAT, ESCALATE, INVESTIGATE
    - **Data Source**: MIMIC-IV-ED (68,936 real patient records)
    
    *This AI agent is for educational and research purposes only. Not for clinical use.*
    """)


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    # Initialize the agent
    initialize_agent()
    
    # Launch Gradio app
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False
    )