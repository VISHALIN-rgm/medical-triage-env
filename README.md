---
title: Medical Triage AI Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏥 Medical Triage Environment
> **Hospital-Grade OpenEnv RL Environment for Emergency Department Triage**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-orange.svg)](https://github.com/meta-pytorch/OpenEnv)
[![Phase2](https://img.shields.io/badge/Phase%202-Passed%20✅-brightgreen.svg)]()

---

## 🏥 Environment Overview

The **Medical Triage Environment** simulates a real hospital emergency department where an AI agent must assess, prioritize, and manage multiple patients under time pressure. Built on the **OpenEnv framework**, it evaluates deep RL agents for clinical decision-making using **real patient data**.

### What Makes This Different

| Feature | Basic Triage | This Environment |
|---------|-------------|-----------------|
| RL Algorithm | Simple Q-table | **Dueling Double DQN + PER** |
| Clinical Scoring | Basic vitals | **NEWS2 + Sepsis-3 + SOFA + SIRS** |
| Patient Data | Synthetic | **68,936 real MIMIC-IV-ED records** |
| State Space | 3 buckets | **18-feature clinical vector** |
| Hard Mode | Static patients | **Real-time vitals deterioration** |
| LLM Role | None | **Clinical reasoning via proxy** |

---

## 💡 Motivation

Emergency departments worldwide face critical challenges:
- **Overcrowding**: 80% of EDs operate at or above capacity
- **Triage errors**: 1 in 10 patients are mis-triaged, costing lives
- **Delayed care**: Every minute of delay increases mortality for critical patients

This environment enables training and evaluation of AI agents that can:
- ✅ Assess patient urgency using clinically validated scoring systems (NEWS2)
- ✅ Detect sepsis risk using Sepsis-3 criteria before it becomes critical
- ✅ Adapt decisions as patient conditions deteriorate in real time
- ✅ Learn optimal triage strategies through deep reinforcement learning
- ✅ Provide explainable clinical reasoning via LLM for every decision

---

## 🧠 AI Architecture

### Dueling Double Deep Q-Network

```
Input: 18-dimensional clinical state vector
         │
    ┌────▼────┐
    │  FC 128 │  ReLU
    └────┬────┘
         │
   ┌─────┴──────┐
   │            │
┌──▼──┐      ┌──▼───┐
│ V(s)│      │A(s,a)│   Dueling streams
│FC 64│      │FC 64 │   Value + Advantage
└──┬──┘      └──┬───┘
   └─────┬──────┘
         │  Q(s,a) = V(s) + A(s,a) - mean(A)
    ┌────▼─────┐
    │ 4 actions│
    └──────────┘
```

### 18-Feature Clinical State Vector

| # | Feature | Clinical Significance |
|---|---------|----------------------|
| 1 | Heart Rate (normalized) | Tachycardia/bradycardia |
| 2 | Oxygen Saturation | Hypoxia severity |
| 3 | Temperature | Fever/hypothermia |
| 4 | Systolic BP | Shock indicator |
| 5 | Diastolic BP | Perfusion pressure |
| 6 | Respiratory Rate | Respiratory distress |
| 7 | NEWS2 Score | Overall acuity |
| 8 | Sepsis Probability | Sepsis-3 risk score |
| 9 | SIRS Count | Inflammatory response |
| 10 | SOFA Score | Organ dysfunction |
| 11 | Deterioration ETA | Time-to-critical |
| 12 | Urgency Score | ESI-based priority |
| 13 | Age | Age-related risk |
| 14 | HR Risk Flag | Binary critical HR |
| 15 | O2 Risk Flag | Binary hypoxia |
| 16 | BP Risk Flag | Binary shock |
| 17 | Temp Risk Flag | Binary fever |
| 18 | Multi-organ Flag | Combined organ risk |

### DQN Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | Dueling Double DQN |
| Experience Replay | Prioritized (PER) |
| Discount Factor γ | 0.99 |
| Learning Rate | 5e-4 |
| Batch Size | 64 |
| Buffer Capacity | 20,000 |
| Epsilon Start/End | 0.95 → 0.02 |

---

## 🩺 Clinical Scoring Systems

### NEWS2 (National Early Warning Score 2)
UK Royal College of Physicians standard:

| NEWS2 | Risk | Action |
|-------|------|--------|
| 0–1 | LOW | DISCHARGE |
| 2–5 | MEDIUM | TREAT |
| ≥6 | HIGH | ESCALATE |

### Sepsis-3 Screening
2+ SIRS criteria triggers sepsis alert → INVESTIGATE or ESCALATE:
- Temperature > 38.3°C or < 36.0°C
- Heart Rate > 90 bpm
- Respiratory Rate > 20 breaths/min
- Systolic BP < 100 mmHg

### SOFA Score
Organ dysfunction estimation from bedside vitals. SOFA ≥ 2 with suspected infection = Sepsis-3.

---

## 🎮 Action Space

| Action | When to Use | Reward |
|--------|-------------|--------|
| **DISCHARGE** | NEWS2 0-1, stable | +10 if correct |
| **TREAT** | NEWS2 2-5, moderate | +10 if correct |
| **ESCALATE** | NEWS2 ≥6, sepsis | +10 if correct |
| **INVESTIGATE** | Confidence < 72% | +5 if appropriate |

### Shaped Reward Function

```python
# Base reward
if action == expected:           reward = +10.0
elif action == "investigate":    reward = +5.0 if confidence < 0.72 else -2.0
elif HIGH patient → DISCHARGE:   reward = -8.0   # dangerous under-treatment
elif LOW patient → ESCALATE:     reward = -5.0   # wasteful over-escalation
else:                            reward = -6.0

# Sepsis early detection bonus
if sepsis_prob > 0.6 and action in ("escalate", "investigate"):
    reward += 3.0

# Hard mode time penalty — delay costs lives
if task == "hard" and risk == "HIGH":
    reward -= 0.5 * step_number
```

---

## 📊 Tasks & Difficulty Levels

### Task 1: Easy (3 patients, max 10 pts)
Static patients with clear clinical signals. Tests basic NEWS2 assessment.

### Task 2: Medium (5 patients, max 20 pts)
Mixed urgency levels with borderline cases. Sepsis screening earns bonus rewards.

### Task 3: Hard (3 deteriorating patients, max 25 pts)
Patient vitals **worsen in real time** each step:

```python
DETERIORATION = {
    "LOW":    {"hr": 0,   "o2":  0.0, "sbp":  0},
    "MEDIUM": {"hr": +4,  "o2": -0.8, "sbp": -4},
    "HIGH":   {"hr": +10, "o2": -2.0, "sbp": -8},
}
```

Delayed treatment makes critical patients harder to save — tests time-critical decision making.

---

## 📦 Real Patient Data — MIMIC-IV-ED

| Field | Details |
|-------|---------|
| **Source** | Beth Israel Deaconess Medical Center, Boston |
| **Years** | 2011–2019 |
| **Records** | 68,936 de-identified ED visits |
| **Access** | Public via Hugging Face Datasets |

| Risk Level | Count | % |
|------------|-------|---|
| LOW (ESI 4-5) | 241 | 0.35% |
| MEDIUM (ESI 3) | 58,085 | 84.3% |
| HIGH (ESI 1-2) | 10,610 | 15.4% |

---

## 🏆 Performance Results

### DQN Agent vs Random Baseline

| Metric | Random | DQN Agent | Improvement |
|--------|--------|-----------|-------------|
| Accuracy | ~25% | **90.9%** | +65.9% |
| Easy | 2.5/10 | **10.0/10** | +300% |
| Medium | 5.0/20 | **20.0/20** | +300% |
| Hard | 6.25/25 | **25.0/25** | +300% |
| **Total** | **~14** | **55.0/55** | **+300%** |

---

## 🚀 Setup & Installation

```bash
git clone https://github.com/VISHALIN-rgm/medical-triage-env
cd medical-triage-env
pip install -r server/requirements.txt

# Set LLM proxy credentials
export API_BASE_URL=https://api.groq.com/openai/v1
export API_KEY=your_key_here
export MODEL_NAME=llama-3.3-70b-versatile

uvicorn inference:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t medical-triage .
docker run -p 7860:7860 -e API_KEY=your_key medical-triage
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Take action → observation + reward |
| `/health` | GET | Health check |
| `/stats` | GET | Agent performance stats |
| `/predict` | POST | Single patient prediction |
| `/sepsis_screen` | POST | Sepsis risk screening |
| `/docs` | GET | Interactive API documentation |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│           HOSPITAL-GRADE MEDICAL TRIAGE AI v50           │
├──────────────────────────────────────────────────────────┤
│  FastAPI REST API  (OpenEnv spec)                        │
│  /reset  /step  /health  /stats  /predict  /sepsis       │
├──────────────────────────────────────────────────────────┤
│  LLM Clinical Reasoning  (LiteLLM Proxy)                 │
├──────────────────────────────────────────────────────────┤
│  Dueling Double DQN  (PyTorch)                           │
│  18-feature state │ PER replay │ Target network          │
├──────────────────────────────────────────────────────────┤
│  Clinical Scoring Engine                                 │
│  NEWS2 │ Sepsis-3 │ SOFA │ SIRS │ Deterioration ETA     │
├──────────────────────────────────────────────────────────┤
│  MIMIC-IV-ED Real Patient Data  (68,936 records)         │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Frameworks Used

| Framework | Purpose |
|-----------|---------|
| **PyTorch 2.0+** | Dueling Double DQN |
| **OpenEnv** | RL environment spec |
| **FastAPI** | REST API server |
| **Pydantic** | Typed data models |
| **OpenAI SDK** | LLM proxy client |
| **HuggingFace Datasets** | MIMIC-IV-ED data |

---

## 🌐 Live Demo

- **HF Space**: https://huggingface.co/spaces/kvishalini/medical-triage-env
- **Live API**: https://kvishalini-medical-triage-env.hf.space
- **API Docs**: https://kvishalini-medical-triage-env.hf.space/docs

---

## 🙏 Acknowledgments

- MIT Lab for Computational Physiology — MIMIC-IV-ED database
- Beth Israel Deaconess Medical Center — Patient data
- UK Royal College of Physicians — NEWS2 standard
- Meta OpenEnv Team — Framework

---

*Built for clinical decision support and AI agent evaluation in emergency medicine 🏥🚀*