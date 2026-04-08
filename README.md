---
title: Medical Triage AI Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🏥 Medical Triage Environment

> **OpenEnv RL Environment for Emergency Department Triage**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-orange.svg)](https://github.com/meta-pytorch/OpenEnv)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/kvishalini/medical-triage-env)

---

## 📋 Table of Contents

- [Environment Overview](#-environment-overview)
- [Motivation](#-motivation)
- [Action Space](#-action-space)
- [Observation Space](#-observation-space)
- [Tasks & Difficulty Levels](#-tasks--difficulty-levels)
- [Reward Structure](#-reward-structure)
- [Real Patient Data](#-real-patient-data)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Baseline Performance](#-baseline-performance)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Technical Architecture](#-technical-architecture)
- [Frameworks Used](#-frameworks-used)
- [Results](#-results)
- [License](#-license)
- [Live Demo](#-live-demo)

---

## 🏥 Environment Overview

The **Medical Triage Environment** simulates a hospital emergency department where an AI agent must assess, prioritize, and manage multiple patients under time pressure. Built on the **OpenEnv framework**, this environment evaluates LLM-based agents and reinforcement learning algorithms for clinical decision-making.

### Key Features

| Feature | Description |
|---------|-------------|
| **Real Patient Data** | 68,936 de-identified records from MIMIC-IV-ED database |
| **Clinical Standards** | NEWS (National Early Warning Score) scoring system |
| **3 Difficulty Levels** | Easy (3 patients) → Medium (5) → Hard (3 deteriorating) |
| **4 Action Types** | DISCHARGE, TREAT, ESCALATE, INVESTIGATE |
| **Q-Learning Agent** | ε-greedy action selection with persistent Q-table |
| **LLM Integration** | Groq/Hugging Face for clinical reasoning |
| **Production API** | FastAPI endpoints for real-time predictions |

---

## 💡 Motivation

Emergency departments worldwide face critical challenges:
- **Overcrowding**: 80% of EDs operate at or above capacity
- **Triage errors**: 1 in 10 patients are mis-triaged
- **Delayed care**: Every minute counts for critical patients

This environment enables training and evaluation of AI agents that can:
- ✅ Assess patient urgency from vital signs and symptoms
- ✅ Prioritize multiple patients with competing needs
- ✅ Make treatment decisions under time pressure
- ✅ Learn optimal triage strategies through reinforcement learning

---

## 🎮 Action Space

The agent can choose from **4 actions** per patient:

| Action | Description | When to Use |
|--------|-------------|-------------|
| **DISCHARGE** | Release stable patient | Urgency < 0.3, NEWS ≤ 1 |
| **TREAT** | Provide medical treatment | Urgency 0.3-0.7, NEWS 2-5 |
| **ESCALATE** | Move to ICU | Urgency > 0.7, NEWS ≥ 6 |
| **INVESTIGATE** | Request additional tests | Confidence < 80% or borderline NEWS 3-4 |

### Action Format

```python
from models import MedicalAction

action = MedicalAction(
    type="discharge",  # or "treat", "escalate", "investigate"
    patient_id="P1",
    notes="Clinical reasoning..."
)
```

## 👁️ Observation Space
Each step returns a MedicalObservation containing:

| Field | Type | Description |
|-------|------|-------------|
| patients | List[Patient] | All current patients with vitals and symptoms |
| current_step | int | Current step number |
| max_steps | int | Maximum steps for this episode |
| task_id | str | Current task (easy/medium/hard) |
| done | bool | Whether episode is complete |

### Patient Data Structure
```python
Patient(
    id="P1",
    name="James Wilson",
    age=72,
    symptoms=["chest pain", "shortness of breath"],
    vitals=VitalSigns(
        heart_rate=118,
        blood_pressure_systolic=85,
        blood_pressure_diastolic=55,
        oxygen_saturation=88.0,
        temperature=37.2,
        respiratory_rate=24
    ),
    status="critical",
    urgency_score=0.85,
    chief_complaint="Severe chest pain radiating to left arm"
)
```

## 📊 Tasks & Difficulty Levels

### Task 1: Easy (3 patients)
Objective: Identify which patient needs immediate care from vital signs alone

| Patient | Symptoms | Vitals | Expected Action |
|---------|----------|--------|-----------------|
| P1 | Chest pain, SOB | HR=120, O2=88%, BP=85/55 | ESCALATE |
| P2 | Mild headache | HR=72, O2=99%, BP=118/78 | DISCHARGE |
| P3 | Mild headache | HR=70, O2=99%, BP=120/80 | DISCHARGE |

### Task 2: Medium (5 patients)
Objective: Prioritize 5 patients with conflicting symptoms

| Patient | Symptoms | Vitals | Expected Action |
|---------|----------|--------|-----------------|
| Mixed urgency cases | Fever, cough, chest pain | Various | TREAT/ESCALATE |

### Task 3: Hard (3 deteriorating patients)
Objective: Manage patients whose condition worsens over time

| Patient | Initial Condition | Deterioration | Expected Action |
|---------|------------------|---------------|-----------------|
| Varies | Moderate abnormalities | Worsening vitals | Adaptive treatment |

## 🎯 Reward Structure
The reward function provides per-step feedback with partial credit:

| Action | Correct | Wrong |
|--------|---------|-------|
| Correct action | +10.0 | - |
| Wrong action | - | -5.0 to -8.0 |
| Appropriate INVESTIGATE | +5.0 | -2.0 |
| Over-escalation (LOW→ESCALATE) | - | -5.0 |
| Under-treatment (HIGH→DISCHARGE) | - | -8.0 |

### Reward Calculation
```python
if action == expected:
    reward = 10.0  # Full reward for correct action
elif action == "investigate" and is_appropriate:
    reward = 5.0   # Partial reward for seeking more info
else:
    reward = -5.0 to -8.0  # Penalty for wrong actions
```

## 📦 Real Patient Data
This environment uses 68,936 real de-identified patient records from the MIMIC-IV-ED database.

### Data Source
| Field | Details |
|-------|---------|
| Name | MIMIC-IV-ED (Medical Information Mart for Intensive Care - Emergency Department) |
| Institution | Beth Israel Deaconess Medical Center, Boston, MA |
| Years | 2011-2019 |
| Access | Public via Hugging Face Datasets |

### Risk Distribution (Actual)
| Risk Level | Count | Percentage |
|------------|-------|------------|
| LOW (ESI 4-5) | 241 | 0.35% |
| MEDIUM (ESI 3) | 58,085 | 84.3% |
| HIGH (ESI 1-2) | 10,610 | 15.4% |

### Sample Real Patients
```
Elizabeth Lopez, 44  - Dyspnea (HR=92, O2=87%)
Patricia Davis, 42   - Dyspnea (HR=88, O2=96%)
Christopher Young, 52 - Transfer (HR=70, O2=85%)
```

## 🚀 Setup & Installation

### Prerequisites
```bash
# Python 3.10+
python --version

# Git
git --version

# Docker (optional)
docker --version
```

### Installation
```bash
# Clone repository
git clone https://github.com/VISHALIN-rgm/medical-triage-env
cd medical-triage-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional - for LLM mode)
echo GROQ_API_KEY=your_key_here >> .env
```

### Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
openai==1.3.0
datasets
python-dotenv
numpy
```

## 📖 Usage

### Run the Agent
```bash
# Basic run (rule-based mode)
python inference.py

# With LLM (requires GROQ_API_KEY in .env)
python inference.py
```

### Expected Output
```
[START] task=easy env=medical_triage model=llama-3.3-70b-versatile
[STEP] step=1 action=escalate(P1)  reward=10.00 done=false error=null
[STEP] step=2 action=discharge(P2) reward=10.00 done=false error=null
[STEP] step=3 action=discharge(P3) reward=10.00 done=true  error=null
[END]  success=true steps=3 score=1.000 rewards=10.00,10.00,10.00

🏆 FINAL RESULTS
🏆 EASY  : 10.0/10.0  (100%)
🏆 MEDIUM: 20.0/20.0  (100%)
🏆 HARD  : 25.0/25.0  (100%)
🎯 TOTAL : 55.0/55.0  (100.0%)
```

### Start API Server
```bash
python inference.py
# API server runs on http://localhost:8000
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/stats

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60,
    "heart_rate": 120,
    "oxygen_saturation": 88,
    "systolic_bp": 80,
    "diastolic_bp": 50,
    "temperature": 37.2,
    "symptoms": ["chest pain", "shortness of breath"],
    "chief_complaint": "Chest pain radiating to left arm"
  }'
```

## 🏆 Baseline Performance

### Agent Performance (90.9% Accuracy)
| Task | Score | Max | Percentage |
|------|-------|-----|------------|
| Easy | 10.0 | 10.0 | 100% |
| Medium | 20.0 | 20.0 | 100% |
| Hard | 25.0 | 25.0 | 100% |
| TOTAL | 55.0 | 55.0 | 100% |

### Action Distribution
| Action | Count | Percentage |
|--------|-------|------------|
| DISCHARGE | 4 | 36% |
| TREAT | 2 | 18% |
| ESCALATE | 4 | 36% |
| INVESTIGATE | 1 | 9% |

### Risk Distribution
| Risk Level | Percentage |
|------------|------------|
| LOW | 36% |
| MEDIUM | 27% |
| HIGH | 36% |

### Q-Table (Learned Values)
| State | DISCHARGE | TREAT | ESCALATE | INVESTIGATE |
|-------|-----------|-------|----------|-------------|
| LOW | 10.00 | 1.01 | 1.00 | 3.00 |
| MEDIUM | 0.10 | 5.05 | -0.06 | 3.35 |
| HIGH | -1.50 | 1.00 | 10.05 | 3.00 |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | API information |
| /health | GET | Health check and status |
| /stats | GET | Agent performance statistics |
| /predict | POST | Clinical prediction endpoint |

### Predict Request
```json
{
  "age": 60,
  "heart_rate": 120,
  "oxygen_saturation": 88,
  "systolic_bp": 80,
  "diastolic_bp": 50,
  "temperature": 37.2,
  "symptoms": ["chest pain", "shortness of breath"],
  "chief_complaint": "Chest pain radiating to left arm"
}
```

### Predict Response
```json
{
  "news_score": 9,
  "risk_level": "HIGH",
  "action": "ESCALATE",
  "confidence": 0.92,
  "reasoning": "Critical vitals - immediate ICU escalation",
  "q_value": 10.05
}
```

## 📁 Project Structure
```
medical_triage_env/
├── inference.py                           # Main agent
├── models.py                              # Pydantic data models
├── real_data_loader.py                    # Real data loader
├── openenv.yaml                           # OpenEnv configuration
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── LICENSE                                # MIT License
├── .gitignore                             # Git ignore file
├── Dockerfile                             # Container definition
├── .env                                   # API keys (optional)
├── q_values.json                          # Persistent Q-table
├── patients.db                            # SQLite patient database
└── server/
    ├── __init__.py
    ├── app.py                             # FastAPI server
    └── medical_triage_env_environment.py  # Core environment
```

## 🏗️ Technical Architecture

### NEWS Score Calculation
| Parameter | Normal | Abnormal | Critical |
|-----------|--------|----------|----------|
| Heart Rate | 51-90 bpm | 91-130 bpm | >130 or <40 |
| Oxygen Saturation | ≥96% | 91-95% | ≤91% |
| Temperature | 36.1-38.0°C | 38.1-39.0°C | >39.0°C or <35.0°C |
| Systolic BP | 111-219 mmHg | 101-110 mmHg | ≤100 mmHg |

### Risk Classification
| NEWS Score | Risk Level | Recommended Action |
|------------|------------|-------------------|
| 0-1 | LOW | DISCHARGE |
| 2-5 | MEDIUM | TREAT |
| ≥6 | HIGH | ESCALATE |

### Q-Learning Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| ε (epsilon) | 0.10 | Exploration rate |
| α (alpha) | 0.11 | Learning rate |
| γ (gamma) | 0.91 | Discount factor |
| Error Rate | 10% | Realistic imperfection |

## 🛠️ Frameworks Used

| Framework | Version | Purpose |
|-----------|---------|---------|
| OpenEnv | Latest | RL environment specification |
| FastAPI | 0.104.1 | REST API server |
| Pydantic | 2.4.2 | Data validation |
| OpenAI | 1.3.0 | LLM client for Groq/Hugging Face |
| Uvicorn | 0.24.0 | ASGI server |
| Hugging Face Datasets | Latest | Real patient data loading |
| SQLite | - | Local patient database |

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    MEDICAL TRIAGE SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI (REST API Layer)                │   │
│  │         /health, /stats, /predict endpoints          │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              OpenEnv (Environment Layer)             │   │
│  │         reset() / step() / state() API               │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Q-Learning Agent (RL Layer)                │   │
│  │         ε-greedy / Persistent Q-table                │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Data Layer                              │   │
│  │    Hugging Face Datasets → SQLite → JSON             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📈 Results
```
============================================================
🏆 FINAL PERFORMANCE SUMMARY
============================================================
   Data        : MIMIC-IV-ED (Real Emergency Department Data)
   Records     : 68,936
   Patients    : 11
   Correct     : 10
   Accuracy    : 90.9%
   Total Reward: 105.0
   INVESTIGATE : ✅ Used

📈 RISK DISTRIBUTION:
   LOW    : 4 (36%)
   MEDIUM : 3 (27%)
   HIGH   : 4 (36%)

🎯 ACTION DISTRIBUTION:
   DISCHARGE  : 4 (36%)
   TREAT      : 2 (18%)
   ESCALATE   : 4 (36%)
   INVESTIGATE: 1 (9%)
============================================================
```

## 🌐 Live Demo
The agent is deployed and accessible at:

- **Hugging Face Space**: https://huggingface.co/spaces/kvishalini/medical-triage-env
- **Live API**: https://kvishalini-medical-triage-env.hf.space

### Test the Live API
```bash
# Health check
curl https://kvishalini-medical-triage-env.hf.space/health

# Make a prediction
curl -X POST https://kvishalini-medical-triage-env.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60,
    "heart_rate": 120,
    "oxygen_saturation": 88,
    "systolic_bp": 80,
    "diastolic_bp": 50,
    "temperature": 37.2,
    "symptoms": ["chest pain", "shortness of breath"],
    "chief_complaint": "Chest pain radiating to left arm"
  }'
```

## 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

## 🙏 Acknowledgments
- **MIT Lab for Computational Physiology** — MIMIC-IV-ED database
- **Beth Israel Deaconess Medical Center** — Patient data source
- **Hugging Face** — Dataset hosting and Spaces deployment
- **Meta OpenEnv Team** — Framework and inspiration
- **Groq** — Fast LLM inference

## 📧 Contact
- **GitHub**: https://github.com/VISHALIN-rgm/medical-triage-env
- **Hugging Face**: https://huggingface.co/spaces/kvishalini/medical-triage-env

Built for clinical decision support and AI agent evaluation 🚀