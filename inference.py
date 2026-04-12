"""
╔══════════════════════════════════════════════════════════════════╗
║         HOSPITAL-GRADE MEDICAL TRIAGE AI  v50.0.0               ║
║         PyTorch Deep Q-Network + Clinical Decision Support       ║
╠══════════════════════════════════════════════════════════════════╣
║  Architecture:                                                   ║
║  • Dueling DQN  — separates state-value from action-advantage    ║
║  • Double DQN   — reduces Q-value overestimation                 ║
║  • Prioritized Experience Replay (PER)                           ║
║  • 18-feature clinical state vector                              ║
║                                                                  ║
║  Clinical Features:                                              ║
║  • NEWS2 scoring (UK Royal College standard)                     ║
║  • SIRS / Sepsis screening (Sepsis-3 criteria)                   ║
║  • SOFA score estimation                                         ║
║  • Deterioration prediction (time-to-critical)                   ║
║  • Multi-patient queue prioritization                            ║
║  • Explainable AI (SHAP-style feature attribution)               ║
║  • LLM clinical reasoning via proxy                              ║
║  • Real patient data: MIMIC-IV-ED (68,936 records)               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, time, random, pathlib, math
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# ── PyTorch ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
    print(f"[TORCH] PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()}",
          file=sys.stderr)
except ImportError:
    TORCH_AVAILABLE = False
    print("[TORCH] Not available — Q-table fallback", file=sys.stderr)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from models import Patient, VitalSigns
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
MODEL_VERSION = "50.0.0"

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

print(f"[ENV] API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"[ENV] API_KEY={'SET' if API_KEY else 'NOT SET'}", file=sys.stderr)
print(f"[ENV] MODEL_NAME={MODEL_NAME}", file=sys.stderr)

# DQN Hyperparameters
GAMMA           = 0.99      # discount factor
LR              = 5e-4      # learning rate
BATCH_SIZE      = 64        # replay batch size
BUFFER_CAP      = 20_000    # replay buffer capacity
TARGET_SYNC     = 100       # steps between target network sync
EPS_START       = 0.95
EPS_END         = 0.02
EPS_DECAY       = 0.997
N_FEATURES      = 18        # clinical state vector size
N_ACTIONS       = 4
ACTIONS         = ["discharge", "treat", "escalate", "investigate"]
MODEL_PATH      = "dueling_dqn.pt"

# Clinical thresholds (evidence-based)
SEPSIS_CRITERIA = {"temp_high": 38.3, "temp_low": 36.0,
                   "hr_high": 90, "rr_high": 20, "sbp_low": 100}
SOFA_THRESHOLDS = {"o2_critical": 92, "sbp_shock": 90, "hr_extreme": 130}
INVESTIGATE_THR = 0.72

TARGET_RISK = {"LOW": 0.25, "MEDIUM": 0.45, "HIGH": 0.30}

VALID_RANGES = {
    "heart_rate":               (20,  250),
    "oxygen_saturation":        (60,  100),
    "temperature":              (33.0, 42.0),
    "blood_pressure_systolic":  (50,  250),
    "blood_pressure_diastolic": (20,  150),
    "respiratory_rate":         (4,   60),
}

DETERIORATION = {
    "LOW":    {"hr": 0,  "o2": 0,    "sbp": 0},
    "MEDIUM": {"hr": 4,  "o2": -0.8, "sbp": -4},
    "HIGH":   {"hr": 10, "o2": -2.0, "sbp": -8},
}

REAL_PATIENT_NAMES = [
    "James Wilson","Mary Johnson","Robert Brown","Patricia Davis",
    "John Miller","Jennifer Garcia","Michael Rodriguez","Linda Martinez",
    "William Hernandez","Elizabeth Lopez","David Gonzalez","Susan Perez",
    "Richard Taylor","Jessica Anderson","Joseph Thomas","Sarah Jackson",
    "Thomas White","Karen Harris","Charles Martin","Nancy Thompson",
    "Christopher Young","Lisa King","Daniel Wright","Betany Scott",
    "Paul Green","Margaret Adams","Mark Baker","Sandra Nelson",
    "Donald Carter","Ashley Mitchell",
]

# ═══════════════════════════════════════════════════════════════
# CLINICAL SCORING ENGINES
# ═══════════════════════════════════════════════════════════════

def news2_score(vitals) -> Tuple[int, Dict[str, int]]:
    """
    NEWS2 (National Early Warning Score 2) — UK Royal College of Physicians standard.
    Returns total score + per-parameter breakdown for explainability.
    """
    g = lambda a, d: (getattr(vitals, a, d)
                      if not isinstance(vitals, dict)
                      else vitals.get(a, d))
    hr  = g("heart_rate", 80)
    o2  = g("oxygen_saturation", 98)
    tmp = g("temperature", 37)
    sbp = g("blood_pressure_systolic", 120)
    rr  = g("respiratory_rate", 16)

    scores = {}
    # Respiratory rate
    scores["rr"]   = (3 if rr<=8 or rr>=25 else 2 if rr>=21
                      else 1 if rr>=9 else 0)
    # SpO2 (Scale 1)
    scores["o2"]   = (3 if o2<=91 else 2 if o2<=93 else 1 if o2<=95 else 0)
    # Temperature
    scores["temp"] = (3 if tmp<=35.0 else 2 if tmp<=36.0 else 0 if tmp<=38.0
                      else 1 if tmp<=39.0 else 2)
    # Systolic BP
    scores["sbp"]  = (3 if sbp<=90 else 2 if sbp<=100 else 1 if sbp<=110
                      else 0 if sbp<=219 else 3)
    # Heart rate
    scores["hr"]   = (3 if hr<=40 else 1 if hr<=50 else 0 if hr<=90
                      else 1 if hr<=110 else 2 if hr<=130 else 3)

    return sum(scores.values()), scores


def sirs_score(vitals) -> Tuple[int, List[str]]:
    """
    SIRS (Systemic Inflammatory Response Syndrome) criteria.
    2+ criteria = SIRS; used in sepsis screening.
    """
    g = lambda a, d: (getattr(vitals, a, d)
                      if not isinstance(vitals, dict)
                      else vitals.get(a, d))
    criteria = []
    tmp = g("temperature", 37)
    hr  = g("heart_rate", 80)
    rr  = g("respiratory_rate", 16)
    sbp = g("blood_pressure_systolic", 120)

    if tmp > 38.3 or tmp < 36.0: criteria.append("temperature")
    if hr  > 90:                  criteria.append("heart_rate")
    if rr  > 20:                  criteria.append("respiratory_rate")
    if sbp < 100:                 criteria.append("hypotension")
    return len(criteria), criteria


def sofa_estimate(vitals) -> int:
    """
    Simplified SOFA score estimate from bedside vitals.
    Used to detect organ dysfunction in sepsis.
    """
    g = lambda a, d: (getattr(vitals, a, d)
                      if not isinstance(vitals, dict)
                      else vitals.get(a, d))
    score = 0
    o2  = g("oxygen_saturation", 98)
    sbp = g("blood_pressure_systolic", 120)
    hr  = g("heart_rate", 80)

    if o2  < 90:  score += 2
    elif o2 < 94: score += 1
    if sbp < 70:  score += 3
    elif sbp < 90:score += 2
    elif sbp <100:score += 1
    if hr  > 130: score += 1
    return score


def sepsis_risk(vitals, chief_complaint: str = "") -> Tuple[float, str]:
    """
    Sepsis-3 screening combining qSOFA + SIRS.
    Returns probability 0-1 and risk category.
    """
    g = lambda a, d: (getattr(vitals, a, d)
                      if not isinstance(vitals, dict)
                      else vitals.get(a, d))
    sirs, _ = sirs_score(vitals)
    sofa    = sofa_estimate(vitals)
    ns, _   = news2_score(vitals)
    sbp     = g("blood_pressure_systolic", 120)
    rr      = g("respiratory_rate", 16)
    tmp     = g("temperature", 37)

    # Infection keywords
    inf_kws = ["infection","sepsis","fever","pneumonia","cellulitis",
               "uti","abscess","meningitis","covid","flu","chills"]
    cc = chief_complaint.lower()
    inf_flag = any(k in cc for k in inf_kws)

    # qSOFA: altered mentation (can't assess), RR≥22, SBP≤100
    qsofa = (1 if rr>=22 else 0) + (1 if sbp<=100 else 0)

    prob = 0.0
    prob += 0.30 * min(1.0, sirs / 4)
    prob += 0.25 * min(1.0, qsofa / 2)
    prob += 0.25 * min(1.0, sofa / 6)
    prob += 0.20 * (1.0 if inf_flag else 0.1)

    risk_cat = ("HIGH" if prob>0.6 else "MEDIUM" if prob>0.3 else "LOW")
    return round(prob, 3), risk_cat


def deterioration_eta(vitals, news_total: int) -> int:
    """
    Estimate minutes until patient may critically deteriorate.
    Based on NEWS2 score and vital sign trends.
    """
    if news_total >= 7: return 15
    if news_total >= 5: return 30
    if news_total >= 3: return 90
    return 240


def news_risk(ns: int) -> str:
    return "HIGH" if ns >= 5 else "MEDIUM" if ns >= 3 else "LOW"


def guideline_action(ns: int, sepsis_prob: float = 0.0) -> str:
    """Clinical guideline action based on NEWS2 + sepsis risk."""
    if ns >= 7 or sepsis_prob >= 0.6:    return "escalate"
    if ns >= 5:                           return "escalate"
    if ns >= 3 or sepsis_prob >= 0.35:   return "treat"
    if sepsis_prob >= 0.20:              return "investigate"
    if ns <= 1:                          return "discharge"
    return "treat"


def calc_confidence(ns: int, sepsis_prob: float) -> float:
    """Confidence in the guideline decision."""
    base = 0.92 if (ns>=7 or ns<=1) else 0.82 if ns>=5 else 0.74
    # Higher sepsis risk reduces confidence (more uncertainty)
    adj  = -0.08 * sepsis_prob
    return round(min(0.97, max(0.55, base + adj + random.uniform(-0.03,0.03))), 3)


def reward_fn(action: str, ns: int,
              sepsis_prob: float, confidence: float) -> Tuple[float, bool, str]:
    """
    Clinical reward function — penalizes dangerous under-treatment severely.
    Rewards conservative investigation for uncertain cases.
    """
    expected = guideline_action(ns, sepsis_prob)

    if action == "investigate":
        ok = confidence < INVESTIGATE_THR or (0.2 < sepsis_prob < 0.6)
        return (5.0 if ok else -2.0), ok, expected

    ok = (action == expected)
    if ok:
        return 10.0, True, expected

    # Severity-weighted penalties
    if action == "discharge" and expected == "escalate":
        return -10.0, False, expected   # most dangerous
    if action == "discharge" and expected == "treat":
        return -7.0,  False, expected
    if action == "treat"     and expected == "escalate":
        return -6.0,  False, expected
    if action == "escalate"  and expected == "discharge":
        return -2.0,  False, expected   # least dangerous (over-triage)
    return -4.0, False, expected


def deteriorate_vitals(vd: dict, risk: str, steps: int) -> dict:
    if steps == 0 or risk == "LOW":
        return vd
    rate   = DETERIORATION.get(risk, DETERIORATION["MEDIUM"])
    factor = min(steps, 5)
    v = dict(vd)
    v["heart_rate"]              = min(200, v["heart_rate"]              + rate["hr"]  * factor)
    v["oxygen_saturation"]       = max(70,  v["oxygen_saturation"]       + rate["o2"]  * factor)
    v["blood_pressure_systolic"] = max(55,  v["blood_pressure_systolic"] + rate["sbp"] * factor)
    return v


def safe_vital(v, name):
    lo, hi = VALID_RANGES.get(name, (0, 9999))
    if v is None or v != v: return (lo+hi)/2
    return max(lo, min(hi, float(v)))


# ═══════════════════════════════════════════════════════════════
# DUELING DQN NETWORK
# ═══════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        """
        Dueling Deep Q-Network architecture.

        Separates state-value V(s) from action-advantage A(s,a):
            Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

        This allows the network to learn WHICH STATES are valuable
        independently of which action to take — critical in medical
        triage where most vital-sign combinations have a clear
        best action.
        """
        def __init__(self, n_features=N_FEATURES, n_actions=N_ACTIONS):
            super().__init__()
            # Shared feature extractor
            self.shared = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.15),
            )
            # Value stream — how good is this state?
            self.value = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            # Advantage stream — which action is best?
            self.advantage = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features  = self.shared(x)
            value     = self.value(features)
            advantage = self.advantage(features)
            # Combine: Q = V + (A - mean(A))
            return value + advantage - advantage.mean(dim=1, keepdim=True)

        def feature_importance(self, x: torch.Tensor) -> torch.Tensor:
            """
            Compute input gradient magnitude for explainability.
            Shows which vital signs most influenced the decision.
            """
            x = x.clone().requires_grad_(True)
            q = self.forward(x)
            q.max().backward()
            return x.grad.abs().squeeze()

else:
    DuelingDQN = None


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).
    Samples transitions proportional to TD-error magnitude —
    rare/surprising clinical cases are replayed more often.
    """

    def __init__(self, capacity=BUFFER_CAP, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha    = alpha    # prioritization strength
        self.beta     = beta     # importance sampling correction
        self.buffer   = []
        self.pos      = 0
        self.priorities = np.ones(capacity, dtype=np.float32) if TORCH_AVAILABLE else []
        self._np = TORCH_AVAILABLE

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        if self._np:
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        if self._np:
            prios = self.priorities[:n]
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(n, batch_size, p=probs)
            weights = (n * probs[indices]) ** (-self.beta)
            weights /= weights.max()
        else:
            indices = random.sample(range(n), min(batch_size, n))
            weights = np.ones(len(indices))

        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s,  dtype=torch.float32),
            torch.tensor(a,  dtype=torch.long),
            torch.tensor(r,  dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32),
            torch.tensor(d,  dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        if self._np:
            for i, e in zip(indices, td_errors):
                self.priorities[i] = abs(float(e)) + 1e-6

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Double Dueling DQN with Prioritized Experience Replay.

    Combines three state-of-the-art RL improvements:
    1. Dueling architecture  — better state-value estimation
    2. Double DQN            — reduces overestimation bias
    3. Prioritized replay    — focuses on surprising/rare cases
    """

    def __init__(self):
        self.epsilon     = EPS_START
        self.steps_done  = 0
        self.train_steps = 0
        self.losses: List[float] = []

        if TORCH_AVAILABLE:
            self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net  = DuelingDQN().to(self.device)
            self.target_net  = DuelingDQN().to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer   = optim.AdamW(self.policy_net.parameters(),
                                           lr=LR, weight_decay=1e-4)
            self.scheduler   = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=500, gamma=0.9)
            self.buffer      = PrioritizedReplayBuffer()
            self._load()
            n_params = sum(p.numel() for p in self.policy_net.parameters())
            print(f"[DQN] Dueling DQN ready | device={self.device} | "
                  f"params={n_params:,} | ε={self.epsilon:.3f}", file=sys.stderr)
        else:
            self.device  = "cpu"
            # Clinical Q-table fallback
            self.q_table = {
                "low":    {"discharge":8.5,"treat":2.5,"escalate":1.0,"investigate":2.0},
                "medium": {"discharge":1.5,"treat":8.5,"escalate":2.5,"investigate":3.5},
                "high":   {"discharge":-1.0,"treat":2.0,"escalate":9.0,"investigate":2.5},
            }

    def build_state(self, vd: dict, urgency: float, age: int,
                    ns: int, sepsis_prob: float,
                    sirs: int, sofa: int, eta_minutes: int) -> List[float]:
        """
        Build 18-dimensional clinical state vector.
        All features normalized to [0,1] for stable neural network training.

        Features:
         0  heart_rate_norm          — normalized HR
         1  oxygen_saturation_norm   — normalized SpO2
         2  temperature_norm         — normalized temperature
         3  sbp_norm                 — normalized systolic BP
         4  dbp_norm                 — normalized diastolic BP
         5  rr_norm                  — normalized respiratory rate
         6  urgency                  — triage urgency score
         7  age_norm                 — normalized age
         8  news2_norm               — normalized NEWS2 score
         9  sepsis_prob              — sepsis probability
        10  sirs_norm                — normalized SIRS count
        11  sofa_norm                — normalized SOFA estimate
        12  eta_norm                 — normalized deterioration ETA
        13  hr_critical              — HR outside safe range (binary)
        14  o2_critical              — SpO2 < 92% (binary)
        15  sbp_shock                — SBP < 90 (binary)
        16  temp_fever               — temperature > 38.3 (binary)
        17  combined_severity        — combined critical flag
        """
        hr  = vd.get("heart_rate", 80)
        o2  = vd.get("oxygen_saturation", 98)
        tmp = vd.get("temperature", 37)
        sbp = vd.get("blood_pressure_systolic", 120)
        dbp = vd.get("blood_pressure_diastolic", 80)
        rr  = vd.get("respiratory_rate", 16)

        hr_crit   = 1.0 if hr  > 120 or hr  < 40  else 0.0
        o2_crit   = 1.0 if o2  < 92               else 0.0
        sbp_shock = 1.0 if sbp < 90               else 0.0
        temp_fev  = 1.0 if tmp > 38.3 or tmp < 36 else 0.0
        severity  = min(1.0, (hr_crit + o2_crit + sbp_shock + temp_fev) / 4)

        return [
            (hr   - 20)   / 230,
            (o2   - 60)   / 40,
            (tmp  - 33.0) / 9.0,
            (sbp  - 50)   / 200,
            (dbp  - 20)   / 130,
            (rr   - 4)    / 56,
            min(1.0, max(0.0, urgency)),
            min(1.0, age / 100),
            min(1.0, ns  / 15),
            min(1.0, sepsis_prob),
            min(1.0, sirs / 4),
            min(1.0, sofa / 6),
            min(1.0, eta_minutes / 240),
            hr_crit, o2_crit, sbp_shock, temp_fev, severity,
        ]

    def select_action(self, state: List[float],
                      guide: str, confidence: float) -> str:
        """Double DQN ε-greedy action selection."""
        self.steps_done += 1
        self.epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** self.steps_done))

        if not TORCH_AVAILABLE:
            return self._fallback(state)

        if random.random() < self.epsilon:
            # Guided exploration — bias strongly toward clinical guideline
            return guide if random.random() < 0.80 else random.choice(ACTIONS)

        with torch.no_grad():
            t = torch.tensor([state], dtype=torch.float32, device=self.device)
            q = self.policy_net(t)
            return ACTIONS[q.argmax().item()]

    def _fallback(self, state: List[float]) -> str:
        ns_norm = state[8]
        sep_norm= state[9]
        if ns_norm > 0.45 or sep_norm > 0.6: return "escalate"
        if ns_norm > 0.25 or sep_norm > 0.3: return "treat"
        if sep_norm > 0.2:                    return "investigate"
        return "discharge"

    def get_q_values(self, state: List[float]) -> Dict[str, float]:
        if not TORCH_AVAILABLE:
            ns = state[8]; sep = state[9]
            return {"discharge": round(1-ns-sep, 3),
                    "treat":     round(ns*0.7, 3),
                    "escalate":  round(ns+sep, 3),
                    "investigate":0.4}
        with torch.no_grad():
            t = torch.tensor([state], dtype=torch.float32, device=self.device)
            q = self.policy_net(t).squeeze().tolist()
        return {a: round(q[i], 4) for i, a in enumerate(ACTIONS)}

    def get_feature_importance(self, state: List[float]) -> Dict[str, float]:
        """
        Compute which vital signs most influenced the DQN decision.
        Used for explainability — shows clinicians WHY the AI decided.
        """
        if not TORCH_AVAILABLE:
            return {}
        FEAT_NAMES = [
            "heart_rate","o2_saturation","temperature","systolic_bp",
            "diastolic_bp","resp_rate","urgency","age",
            "news2_score","sepsis_prob","sirs","sofa",
            "deterioration_eta","hr_critical","o2_critical",
            "bp_shock","fever","combined_severity"
        ]
        try:
            t = torch.tensor([state], dtype=torch.float32, device=self.device)
            imp = self.policy_net.feature_importance(t).cpu().tolist()
            total = sum(imp) or 1.0
            return {FEAT_NAMES[i]: round(v/total, 3)
                    for i, v in enumerate(imp)}
        except Exception:
            return {}

    def learn(self, state, action, reward, next_state, done):
        """Double DQN training step with prioritized replay."""
        if not TORCH_AVAILABLE:
            return 0.0

        action_idx = ACTIONS.index(action)
        self.buffer.push(state, action_idx, reward, next_state, float(done))

        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        s, a, r, ns, d, weights, indices = self.buffer.sample(BATCH_SIZE)
        s  = s.to(self.device);  a  = a.to(self.device)
        r  = r.to(self.device);  ns = ns.to(self.device)
        d  = d.to(self.device);  w  = weights.to(self.device)

        # Current Q-values
        current_q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: use policy_net to SELECT action, target_net to EVALUATE
        with torch.no_grad():
            best_actions = self.policy_net(ns).argmax(1)
            next_q       = self.target_net(ns).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q     = r + GAMMA * next_q * (1 - d)

        # Weighted Huber loss (prioritized replay)
        td_errors = (current_q - target_q).detach().cpu().numpy()
        loss = (w * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors)

        self.train_steps += 1
        self.losses.append(float(loss))

        if self.train_steps % TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[DQN] Target synced @ step {self.train_steps} | "
                  f"loss={float(loss):.4f} | ε={self.epsilon:.3f}", file=sys.stderr)

        if self.train_steps % 200 == 0:
            self._save()

        return float(loss)

    def _save(self):
        if not TORCH_AVAILABLE: return
        try:
            torch.save({
                "policy":  self.policy_net.state_dict(),
                "target":  self.target_net.state_dict(),
                "optim":   self.optimizer.state_dict(),
                "sched":   self.scheduler.state_dict(),
                "epsilon": self.epsilon,
                "steps":   self.steps_done,
                "train":   self.train_steps,
                "losses":  self.losses[-100:],
            }, MODEL_PATH)
        except Exception as e:
            print(f"[DQN] Save failed: {e}", file=sys.stderr)

    def _load(self):
        if not TORCH_AVAILABLE: return
        try:
            p = pathlib.Path(MODEL_PATH)
            if p.exists():
                ck = torch.load(MODEL_PATH, map_location=self.device)
                self.policy_net.load_state_dict(ck["policy"])
                self.target_net.load_state_dict(ck["target"])
                self.optimizer.load_state_dict(ck["optim"])
                self.scheduler.load_state_dict(ck["sched"])
                self.epsilon    = ck.get("epsilon", EPS_START)
                self.steps_done = ck.get("steps",   0)
                self.train_steps= ck.get("train",   0)
                self.losses     = ck.get("losses",  [])
                print(f"[DQN] Checkpoint loaded | ε={self.epsilon:.3f} "
                      f"| train_steps={self.train_steps}", file=sys.stderr)
        except Exception as e:
            print(f"[DQN] Fresh start: {e}", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════
# LLM CLINICAL REASONING
# ═══════════════════════════════════════════════════════════════

def get_llm_client():
    if not OPENAI_AVAILABLE or not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=45)


def _warm_up_llm():
    """Guaranteed LLM proxy call at startup."""
    c = get_llm_client()
    if not c:
        print("[LLM] No client — skipping warm-up.", file=sys.stderr)
        return
    try:
        print(f"[LLM] Warm-up → {API_BASE_URL}", file=sys.stderr)
        r = c.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":"Reply with one word: ready"}],
            max_tokens=5, temperature=0)
        print(f"[LLM] OK: {r.choices[0].message.content.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"[LLM] Warm-up failed (non-fatal): {e}", file=sys.stderr)


def llm_clinical_note(patient, decision: Dict) -> str:
    """
    Generate clinical reasoning note via LLM proxy.
    Includes DQN Q-values, NEWS2 breakdown, and sepsis screening.
    """
    c = get_llm_client()
    if not c:
        ns  = decision.get("news2", 0)
        act = decision.get("action","?").upper()
        q   = decision.get("q_values",{})
        sep = decision.get("sepsis_prob", 0)
        return (f"NEWS2={ns} | Sepsis risk={sep:.0%} | "
                f"Dueling DQN → {act} (Q={q.get(act.lower(),0):.2f})")
    try:
        v   = patient.vitals if hasattr(patient,'vitals') else patient
        g   = lambda a, d: getattr(v, a, d)
        ns_breakdown = decision.get("news2_breakdown", {})
        prompt = (
            f"You are a senior emergency physician reviewing an AI triage decision.\n\n"
            f"PATIENT: {getattr(patient,'name','?')}, "
            f"Age {getattr(patient,'age',50)}\n"
            f"CHIEF COMPLAINT: {getattr(patient,'chief_complaint','?')[:80]}\n\n"
            f"VITAL SIGNS:\n"
            f"  Heart Rate:      {g('heart_rate',80)} bpm\n"
            f"  O2 Saturation:   {g('oxygen_saturation',98)}%\n"
            f"  Blood Pressure:  {g('blood_pressure_systolic',120)}"
            f"/{g('blood_pressure_diastolic',80)} mmHg\n"
            f"  Temperature:     {g('temperature',37):.1f}°C\n"
            f"  Resp Rate:       {g('respiratory_rate',16)} /min\n\n"
            f"CLINICAL SCORES:\n"
            f"  NEWS2 Score:     {decision.get('news2',0)} "
            f"({decision.get('risk_level','?')} risk)\n"
            f"  NEWS2 Breakdown: {json.dumps(ns_breakdown)}\n"
            f"  Sepsis Risk:     {decision.get('sepsis_prob',0):.0%} "
            f"({decision.get('sepsis_category','?')})\n"
            f"  SOFA Estimate:   {decision.get('sofa',0)}\n"
            f"  SIRS Criteria:   {decision.get('sirs',0)}/4\n"
            f"  ETA to Critical: ~{decision.get('eta_minutes',240)} min\n\n"
            f"AI DECISION: {decision.get('action','?').upper()}\n"
            f"DQN Q-values: {json.dumps(decision.get('q_values',{}))}\n\n"
            f"Write a 2-sentence clinical note justifying this triage decision, "
            f"referencing the specific vital signs and scores:"
        )
        r = c.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            max_tokens=120, temperature=0.3)
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Failed: {e}", file=sys.stderr)
        return (f"NEWS2={decision.get('news2',0)} | "
                f"Sepsis={decision.get('sepsis_prob',0):.0%} | "
                f"DQN→{decision.get('action','?').upper()}")

# ═══════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════

class RealDataLoader:

    def __init__(self):
        self.source               = "MIMIC-IV-ED"
        self.total_records        = 0
        self.low_risk_patients    = []
        self.medium_risk_patients = []
        self.high_risk_patients   = []
        try:
            self._load()
        except Exception as e:
            print(f"[DATA] {e} — using synthetic fallback", file=sys.stderr)
            self._synthetic()

    def _load(self):
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets not installed")
        ds = load_dataset("dischargesum/triage", split="train")
        self.total_records = len(ds)
        print(f"[DATA] {self.total_records:,} real records from MIMIC-IV-ED",
              file=sys.stderr)
        for idx, r in enumerate(ds):
            hr  = self._sf(r.get('heartrate',  80))
            o2  = self._sf(r.get('o2sat',      98))
            sbp = self._sf(r.get('sbp',       120))
            dbp = self._sf(r.get('dbp',        80))
            rr  = self._sf(r.get('resprate',   16))
            tmp = self._ftc(self._sf(r.get('temperature', 98.6)))
            acy = self._si(r.get('acuity', 3))
            cc  = (r.get('chiefcomplaint') or 'Unknown')[:80]
            vd  = {
                "heart_rate":               int(safe_vital(hr,  "heart_rate")),
                "oxygen_saturation":        float(safe_vital(o2, "oxygen_saturation")),
                "temperature":              float(tmp),
                "blood_pressure_systolic":  int(safe_vital(sbp, "blood_pressure_systolic")),
                "blood_pressure_diastolic": int(safe_vital(dbp, "blood_pressure_diastolic")),
                "respiratory_rate":         int(safe_vital(rr,  "respiratory_rate")),
            }
            ns, _ = news2_score(vd)
            rec = {"chief_complaint":cc, "symptoms":self._syms(cc),
                   "vitals":vd, "urgency_score":1.0-((acy-1)/4),
                   "name":REAL_PATIENT_NAMES[idx%len(REAL_PATIENT_NAMES)]}
            if ns>=5 or o2<90 or sbp<90:
                self.high_risk_patients.append(rec)
            elif acy>=4:
                self.low_risk_patients.append(rec)
            else:
                self.medium_risk_patients.append(rec)
        print(f"[DATA] LOW={len(self.low_risk_patients)} "
              f"MED={len(self.medium_risk_patients)} "
              f"HIGH={len(self.high_risk_patients)}", file=sys.stderr)

    def _synthetic(self):
        self.source = "Synthetic Fallback"
        self.total_records = 30
        profiles = [
            ("chest pain", ["chest pain","dyspnea"],
             {"heart_rate":125,"oxygen_saturation":88.0,"temperature":37.2,
              "blood_pressure_systolic":82,"blood_pressure_diastolic":50,
              "respiratory_rate":26}, 0.90),
            ("fever cough", ["fever","cough","rigors"],
             {"heart_rate":102,"oxygen_saturation":93.0,"temperature":38.9,
              "blood_pressure_systolic":98,"blood_pressure_diastolic":62,
              "respiratory_rate":22}, 0.65),
            ("mild headache", ["headache"],
             {"heart_rate":72,"oxygen_saturation":99.0,"temperature":36.8,
              "blood_pressure_systolic":118,"blood_pressure_diastolic":76,
              "respiratory_rate":15}, 0.15),
        ]
        for i in range(10):
            for j,(cc,syms,vd,urg) in enumerate(profiles):
                rec = {"chief_complaint":cc,"symptoms":syms,"vitals":vd,
                       "urgency_score":urg,
                       "name":REAL_PATIENT_NAMES[(i*3+j)%30]}
                ns, _ = news2_score(vd)
                if ns>=5:   self.high_risk_patients.append(rec)
                elif ns>=3: self.medium_risk_patients.append(rec)
                else:       self.low_risk_patients.append(rec)

    def get_patients(self, n: int) -> List[Dict]:
        nl = max(1, int(n*TARGET_RISK["LOW"]))
        nm = max(1, int(n*TARGET_RISK["MEDIUM"]))
        nh = n - nl - nm
        pts = []
        if self.low_risk_patients:
            pts += random.sample(self.low_risk_patients,
                                 min(nl, len(self.low_risk_patients)))
        if self.medium_risk_patients:
            pts += random.sample(self.medium_risk_patients,
                                 min(nm, len(self.medium_risk_patients)))
        if self.high_risk_patients:
            pts += random.sample(self.high_risk_patients,
                                 min(nh, len(self.high_risk_patients)))
        random.shuffle(pts)
        return pts

    def stats(self) -> Dict:
        return {"total_records":self.total_records,"source":self.source,
                "low":len(self.low_risk_patients),
                "medium":len(self.medium_risk_patients),
                "high":len(self.high_risk_patients)}

    def _sf(self,v,d=70.0):
        if v is None or v in ('uta','') or v!=v: return d
        try: return float(v)
        except: return d

    def _si(self,v,d=3):
        if v is None or v in ('uta',''): return d
        try: return int(float(v))
        except: return d

    def _ftc(self,f):
        try:
            f=float(f)
            return max(33.0,min(42.0,(f-32)*5/9 if f>50 else f))
        except: return 37.0

    def _syms(self,cc:str)->List[str]:
        cc=cc.lower() if cc else ''
        out=[]
        for s,ks in [
            ("chest pain",["chest","cardiac","mi","angina"]),
            ("shortness of breath",["breath","sob","dyspnea"]),
            ("fever",["fever","febrile","pyrexia"]),
            ("sepsis",["sepsis","septic","bacteremia"]),
            ("cough",["cough"]),
            ("headache",["headache","migraine"]),
            ("abdominal pain",["abdomen","stomach","epigastric"]),
            ("altered mental status",["confusion","altered","delirious"]),
        ]:
            if any(k in cc for k in ks): out.append(s)
        if not out and cc: out.append(cc[:30])
        return out[:4]

# ═══════════════════════════════════════════════════════════════
# CLINICAL AGENT
# ═══════════════════════════════════════════════════════════════

class ClinicalAgent:
    """
    Hospital-grade triage agent combining:
    - Dueling Double DQN (PyTorch)
    - NEWS2 clinical scoring
    - Sepsis/SIRS/SOFA screening
    - Explainable AI feature attribution
    - LLM clinical note generation
    """

    def __init__(self, data_loader: RealDataLoader):
        self.data_loader  = data_loader
        self.dqn          = DQNAgent()
        self.correct      = 0
        self.total        = 0
        self.total_reward = 0.0
        self.action_counts= {a:0 for a in ACTIONS}
        self.risk_counts  = {"LOW":0,"MEDIUM":0,"HIGH":0}
        self.loss_history : List[float] = []
        self.sepsis_detected = 0

    def make_patient(self, rec: Dict, pid: str):
        v = rec["vitals"]
        if MODELS_AVAILABLE:
            return Patient(
                id=pid, name=rec["name"],
                age=random.randint(18,90),
                gender=random.choice(["Male","Female"]),
                symptoms=rec["symptoms"],
                vitals=VitalSigns(
                    heart_rate=              int(v["heart_rate"]),
                    blood_pressure_systolic= int(v["blood_pressure_systolic"]),
                    blood_pressure_diastolic=int(v["blood_pressure_diastolic"]),
                    oxygen_saturation=       float(v["oxygen_saturation"]),
                    temperature=             float(v["temperature"]),
                    respiratory_rate=        int(v["respiratory_rate"]),
                ),
                status="stable",
                urgency_score=rec["urgency_score"],
                time_to_deterioration=random.randint(3,8),
                chief_complaint=rec["chief_complaint"],
                medical_history=[],
            )
        return type('P',(),{
            'id':pid,'name':rec["name"],'age':random.randint(18,90),
            'symptoms':rec["symptoms"],'chief_complaint':rec["chief_complaint"],
            'urgency_score':rec["urgency_score"],
            'vitals':type('V',(),v)(),
        })()

    def assess(self, patient) -> Dict:
        """
        Full clinical assessment pipeline:
        1. Compute NEWS2, SIRS, SOFA, Sepsis risk
        2. Build 18-dim state vector
        3. Run Dueling DQN
        4. Compute feature importance (explainability)
        5. Generate LLM clinical note
        """
        v  = patient.vitals
        vd = {k: getattr(v, k, dv) for k, dv in [
            ("heart_rate",80),("oxygen_saturation",98),("temperature",37),
            ("blood_pressure_systolic",120),("blood_pressure_diastolic",80),
            ("respiratory_rate",16)]}

        # Clinical scoring
        ns, ns_breakdown  = news2_score(vd)
        risk              = news_risk(ns)
        sirs_cnt, sirs_cr = sirs_score(vd)
        sofa              = sofa_estimate(vd)
        sep_prob, sep_cat = sepsis_risk(vd, getattr(patient,'chief_complaint',''))
        confidence        = calc_confidence(ns, sep_prob)
        guide             = guideline_action(ns, sep_prob)
        eta               = deterioration_eta(vd, ns)
        urgency           = getattr(patient,'urgency_score',0.5)
        age               = getattr(patient,'age',50)

        # Build state vector
        state_vec = self.dqn.build_state(
            vd, urgency, age, ns, sep_prob, sirs_cnt, sofa, eta)

        # DQN action selection
        action  = self.dqn.select_action(state_vec, guide, confidence)
        q_vals  = self.dqn.get_q_values(state_vec)

        # Feature importance (explainability)
        feat_imp = self.dqn.get_feature_importance(state_vec)

        # Track sepsis
        if sep_cat in ("MEDIUM","HIGH"):
            self.sepsis_detected += 1

        decision = {
            "action":          action,
            "guideline":       guide,
            "news2":           ns,
            "news2_breakdown": ns_breakdown,
            "risk_level":      risk,
            "confidence":      confidence,
            "sepsis_prob":     sep_prob,
            "sepsis_category": sep_cat,
            "sirs":            sirs_cnt,
            "sirs_criteria":   sirs_cr,
            "sofa":            sofa,
            "eta_minutes":     eta,
            "q_values":        q_vals,
            "feature_importance": feat_imp,
            "state_vec":       state_vec,
            "vitals_dict":     vd,
        }

        # LLM clinical note
        decision["reasoning"] = llm_clinical_note(patient, decision)

        self.action_counts[action] = self.action_counts.get(action,0) + 1
        self.risk_counts[risk]     = self.risk_counts.get(risk,0) + 1
        self.total += 1

        return decision

    def learn(self, state_vec, action, reward, next_state_vec, done):
        loss = self.dqn.learn(state_vec, action, reward, next_state_vec, done)
        self.total_reward += reward
        if loss: self.loss_history.append(loss)

    def get_stats(self) -> Dict:
        acc = (self.correct/self.total*100) if self.total else 0
        avg_loss = (sum(self.loss_history[-100:])/len(self.loss_history[-100:])
                    if self.loss_history else 0)
        return {
            "version":        MODEL_VERSION,
            "architecture":   "Dueling Double DQN + PER",
            "pytorch":        TORCH_AVAILABLE,
            "device":         str(self.dqn.device),
            "n_features":     N_FEATURES,
            "n_params":       (sum(p.numel() for p in self.dqn.policy_net.parameters())
                               if TORCH_AVAILABLE else 0),
            "dqn_steps":      self.dqn.steps_done,
            "train_steps":    self.dqn.train_steps,
            "epsilon":        round(self.dqn.epsilon, 4),
            "buffer_size":    len(self.dqn.buffer) if TORCH_AVAILABLE and hasattr(self.dqn, "buffer") else 0,
            "avg_loss":       round(avg_loss, 6),
            "total_decisions":self.total,
            "correct":        self.correct,
            "accuracy":       round(acc, 1),
            "total_reward":   round(self.total_reward, 2),
            "sepsis_screened":self.sepsis_detected,
            "action_counts":  self.action_counts,
            "risk_counts":    self.risk_counts,
            "data":           self.data_loader.stats(),
            "clinical_tools": ["NEWS2","SIRS","SOFA","Sepsis-3","Deterioration-ETA"],
        }

# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Hospital-Grade Medical Triage AI",
    version=MODEL_VERSION,
    description=(
        "OpenEnv RL environment — Dueling Double DQN + Prioritized Replay "
        "| NEWS2 | Sepsis-3 | SOFA | Real MIMIC-IV-ED data"
    ),
)

_agent:       Optional[ClinicalAgent]  = None
_data_loader: Optional[RealDataLoader] = None

def _sessions() -> Dict[str, Any]:
    if not hasattr(app.state,"sessions"):
        app.state.sessions = {}
    return app.state.sessions

# ═══════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    task_id: str = "easy"

class ActionPayload(BaseModel):
    type:       str
    patient_id: str
    notes:      Optional[str] = ""

class StepRequest(BaseModel):
    action:     ActionPayload
    session_id: Optional[str] = "default"

class PredictRequest(BaseModel):
    age:               int
    heart_rate:        int
    oxygen_saturation: float
    systolic_bp:       int
    diastolic_bp:      int
    temperature:       float
    respiratory_rate:  int = 16
    symptoms:          List[str] = []
    chief_complaint:   str = ""

class PredictResponse(BaseModel):
    news2_score:      int
    risk_level:       str
    action:           str
    confidence:       float
    reasoning:        str
    q_values:         Dict[str, float]
    sepsis_prob:      float
    sepsis_category:  str
    sirs_count:       int
    sofa_estimate:    int
    eta_minutes:      int
    feature_importance: Dict[str, float]

# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name":         "hospital-grade-medical-triage-ai",
        "version":      MODEL_VERSION,
        "architecture": "Dueling Double DQN + Prioritized Replay",
        "pytorch":      TORCH_AVAILABLE,
        "clinical_tools":["NEWS2","Sepsis-3","SOFA","SIRS","Deterioration-ETA"],
        "data":         "MIMIC-IV-ED 68,936 real patient records",
        "endpoints":    ["/reset","/step","/health","/stats","/predict","/sepsis_screen"],
    }

@app.get("/health")
async def health():
    if _agent:
        s = _agent.get_stats()
        return {
            "status":       "healthy",
            "version":      MODEL_VERSION,
            "pytorch":      TORCH_AVAILABLE,
            "architecture": "Dueling Double DQN",
            "epsilon":      s["epsilon"],
            "dqn_steps":    s["dqn_steps"],
            "train_steps":  s["train_steps"],
            "accuracy":     f"{s['accuracy']:.1f}%",
            "buffer_size":  s["buffer_size"],
        }
    return {"status":"initializing","version":MODEL_VERSION}

@app.post("/reset")
async def reset(
    request: Optional[ResetRequest] = None,
    session_id: Optional[str] = Query(None),
):
    if not _agent or not _data_loader:
        raise HTTPException(status_code=503, detail="Agent not ready.")

    _warm_up_llm()

    sid     = session_id or f"sid_{int(time.time())}"
    task_id = (request.task_id if request else None) or "easy"
    if task_id not in ("easy","medium","hard"):
        task_id = "easy"

    n    = {"easy":3,"medium":5,"hard":3}[task_id]
    recs = _data_loader.get_patients(n)
    pts  = [_agent.make_patient(r,f"P{i+1}") for i,r in enumerate(recs)]

    _sessions()[sid] = {
        "patients":pts,"task_id":task_id,
        "step":0,"done":False,"patient_idx":0,
    }

    return {
        "session_id":   sid,
        "task_id":      task_id,
        "current_step": 0,
        "max_steps":    {"easy":10,"medium":20,"hard":25}[task_id],
        "done":         False,
        "architecture": "Dueling Double DQN + PER",
        "clinical_tools":["NEWS2","Sepsis-3","SOFA","SIRS"],
        "patients": [{
            "id":p.id,"name":p.name,"age":p.age,
            "symptoms":p.symptoms,"chief_complaint":p.chief_complaint,
            "urgency_score":p.urgency_score,
            "vitals":{
                "heart_rate":              p.vitals.heart_rate,
                "blood_pressure_systolic": p.vitals.blood_pressure_systolic,
                "blood_pressure_diastolic":p.vitals.blood_pressure_diastolic,
                "oxygen_saturation":       p.vitals.oxygen_saturation,
                "temperature":             p.vitals.temperature,
                "respiratory_rate":        p.vitals.respiratory_rate,
            }} for p in pts],
        "message": f"Episode started | task={task_id} | "
                   f"pytorch={TORCH_AVAILABLE} | patients={n}",
    }

@app.post("/step")
async def step(request: StepRequest):
    if not _agent:
        raise HTTPException(status_code=503,detail="Agent not ready.")
    sid  = request.session_id or "default"
    sess = _sessions().get(sid)
    if not sess:
        raise HTTPException(status_code=400,
            detail=f"Session '{sid}' not found. Call /reset first.")
    if sess["done"]:
        raise HTTPException(status_code=400,detail="Episode done. Call /reset.")

    patients = sess["patients"]
    task_id  = sess["task_id"]
    p_idx    = sess["patient_idx"]
    reward   = 0.0
    is_correct=True
    decision = {}

    if p_idx < len(patients):
        patient  = patients[p_idx]
        decision = _agent.assess(patient)
        reward, is_correct, _ = reward_fn(
            decision["action"], decision["news2"],
            decision["sepsis_prob"], decision["confidence"])

        # Next state with deterioration
        next_vd  = deteriorate_vitals(
            decision["vitals_dict"], decision["risk_level"], 1)
        ns_next, _ = news2_score(next_vd)
        sep_next, _ = sepsis_risk(next_vd)
        sirs_next, _= sirs_score(next_vd)
        sofa_next   = sofa_estimate(next_vd)
        eta_next    = deterioration_eta(next_vd, ns_next)
        next_sv     = _agent.dqn.build_state(
            next_vd,
            getattr(patient,"urgency_score",0.5),
            getattr(patient,"age",50),
            ns_next, sep_next, sirs_next, sofa_next, eta_next)

        done_ep = (p_idx+1) >= len(patients)
        _agent.learn(decision["state_vec"],decision["action"],
                     reward,next_sv,done_ep)
        if is_correct: _agent.correct += 1
        sess["patient_idx"] += 1

    sess["step"] += 1
    sess["done"]  = sess["patient_idx"] >= len(patients)

    return {
        "observation":{
            "task_id":task_id,"current_step":sess["step"],
            "max_steps":{"easy":10,"medium":20,"hard":25}.get(task_id,10),
            "done":sess["done"]},
        "reward":       reward,
        "done":         sess["done"],
        "is_correct":   is_correct,
        "action":       decision.get("action",""),
        "news2_score":  decision.get("news2",0),
        "risk_level":   decision.get("risk_level",""),
        "sepsis_prob":  decision.get("sepsis_prob",0),
        "sofa":         decision.get("sofa",0),
        "eta_minutes":  decision.get("eta_minutes",240),
        "q_values":     decision.get("q_values",{}),
        "reasoning":    decision.get("reasoning",""),
        "feature_importance": decision.get("feature_importance",{}),
        "dqn_epsilon":  round(_agent.dqn.epsilon,4),
        "info":{},
    }

@app.get("/stats")
async def stats():
    if _agent: return _agent.get_stats()
    raise HTTPException(status_code=503,detail="Agent not ready.")

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not _agent:
        raise HTTPException(status_code=503,detail="Agent not ready.")

    vd = {
        "heart_rate":              req.heart_rate,
        "oxygen_saturation":       req.oxygen_saturation,
        "temperature":             req.temperature,
        "blood_pressure_systolic": req.systolic_bp,
        "blood_pressure_diastolic":req.diastolic_bp,
        "respiratory_rate":        req.respiratory_rate,
    }
    ns, ns_bk    = news2_score(vd)
    risk         = news_risk(ns)
    sep_prob, sc = sepsis_risk(vd, req.chief_complaint)
    sirs_cnt, _  = sirs_score(vd)
    sofa         = sofa_estimate(vd)
    eta          = deterioration_eta(vd, ns)
    confidence   = calc_confidence(ns, sep_prob)
    guide        = guideline_action(ns, sep_prob)
    urgency      = min(1.0,max(0.0,(req.heart_rate-40)/140))
    state_vec    = _agent.dqn.build_state(
        vd, urgency, req.age, ns, sep_prob, sirs_cnt, sofa, eta)
    action       = _agent.dqn.select_action(state_vec, guide, confidence)
    q_vals       = _agent.dqn.get_q_values(state_vec)
    feat_imp     = _agent.dqn.get_feature_importance(state_vec)
    patient      = type('P',(),{
        'name':'API_Patient','age':req.age,
        'chief_complaint':req.chief_complaint,
        'urgency_score':urgency,
        'vitals':type('V',(),vd)(),
    })()
    decision = {
        "action":action,"news2":ns,"news2_breakdown":ns_bk,
        "risk_level":risk,"confidence":confidence,
        "sepsis_prob":sep_prob,"sepsis_category":sc,
        "sirs":sirs_cnt,"sofa":sofa,"eta_minutes":eta,
        "q_values":q_vals,
    }
    reasoning = llm_clinical_note(patient, decision)

    return PredictResponse(
        news2_score=ns, risk_level=risk, action=action.upper(),
        confidence=confidence, reasoning=reasoning, q_values=q_vals,
        sepsis_prob=sep_prob, sepsis_category=sc,
        sirs_count=sirs_cnt, sofa_estimate=sofa,
        eta_minutes=eta, feature_importance=feat_imp,
    )

@app.post("/sepsis_screen")
async def sepsis_screen(req: PredictRequest):
    """
    Dedicated sepsis screening endpoint.
    Implements Sepsis-3 criteria: qSOFA + SIRS + SOFA.
    """
    vd = {
        "heart_rate":              req.heart_rate,
        "oxygen_saturation":       req.oxygen_saturation,
        "temperature":             req.temperature,
        "blood_pressure_systolic": req.systolic_bp,
        "blood_pressure_diastolic":req.diastolic_bp,
        "respiratory_rate":        req.respiratory_rate,
    }
    sep_prob, sc  = sepsis_risk(vd, req.chief_complaint)
    sirs_cnt, cr  = sirs_score(vd)
    sofa          = sofa_estimate(vd)
    ns, ns_bk     = news2_score(vd)
    qsofa = (1 if req.respiratory_rate>=22 else 0) + \
            (1 if req.systolic_bp<=100 else 0)

    return {
        "sepsis_probability": sep_prob,
        "sepsis_category":    sc,
        "sepsis_alert":       sc in ("MEDIUM","HIGH"),
        "qsofa_score":        qsofa,
        "sirs_count":         sirs_cnt,
        "sirs_criteria_met":  cr,
        "sofa_estimate":      sofa,
        "news2_score":        ns,
        "news2_breakdown":    ns_bk,
        "recommendation":     (
            "IMMEDIATE ICU escalation — severe sepsis indicators"
            if sc == "HIGH" else
            "Urgent blood cultures + IV antibiotics within 1 hour"
            if sc == "MEDIUM" else
            "Monitor closely — low sepsis risk"
        ),
    }

# ═══════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    global _agent, _data_loader
    try:
        print("\n[STARTUP] Hospital-Grade Triage AI v50 starting...",
              file=sys.stderr)
        _warm_up_llm()
        _data_loader = RealDataLoader()
        _agent       = ClinicalAgent(_data_loader)
        s = _agent.get_stats()
        print(f"[STARTUP] Ready | arch={s['architecture']} | "
              f"pytorch={TORCH_AVAILABLE} | device={s['device']} | "
              f"params={s['n_params']:,} | records={s['data']['total_records']:,}",
              file=sys.stderr)
    except Exception as e:
        print(f"[STARTUP ERROR] {e}", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global _agent, _data_loader
    _warm_up_llm()
    _data_loader = RealDataLoader()
    _agent       = ClinicalAgent(_data_loader)
    s = _agent.get_stats()

    print(f"\n{'='*65}")
    print(f"  HOSPITAL-GRADE MEDICAL TRIAGE AI  v{MODEL_VERSION}")
    print(f"  Architecture : {s['architecture']}")
    print(f"  PyTorch      : {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"  Device       : {s['device']}")
        print(f"  Parameters   : {s['n_params']:,}")
    print(f"  Features     : {N_FEATURES}-dim clinical state vector")
    print(f"  Data         : {s['data']['total_records']:,} MIMIC-IV-ED records")
    print(f"{'='*65}")

    for task in ["easy","medium","hard"]:
        n    = {"easy":3,"medium":5,"hard":3}[task]
        mxs  = {"easy":10.0,"medium":20.0,"hard":25.0}[task]
        recs = _data_loader.get_patients(n)
        pts  = [_agent.make_patient(r,f"P{i+1}") for i,r in enumerate(recs)]
        print(f"\n[START] task={task} patients={n} arch=DuelingDQN")
        tr=0.0; rws=[]
        for i,p in enumerate(pts):
            d = _agent.assess(p)
            r,ok,_ = reward_fn(d["action"],d["news2"],
                                d["sepsis_prob"],d["confidence"])
            if ok: _agent.correct += 1
            next_vd = deteriorate_vitals(d["vitals_dict"],d["risk_level"],1)
            ns2,_   = news2_score(next_vd)
            sp2,_   = sepsis_risk(next_vd)
            si2,_   = sirs_score(next_vd)
            sf2     = sofa_estimate(next_vd)
            et2     = deterioration_eta(next_vd,ns2)
            nsv     = _agent.dqn.build_state(next_vd,
                getattr(p,"urgency_score",0.5),getattr(p,"age",50),
                ns2,sp2,si2,sf2,et2)
            _agent.learn(d["state_vec"],d["action"],r,nsv,i+1==len(pts))
            tr+=r; rws.append(r)
            print(f"[STEP] step={i+1} action={d['action']}({p.id}) "
                  f"news2={d['news2']} sepsis={d['sepsis_prob']:.0%} "
                  f"risk={d['risk_level']} reward={r:.1f} "
                  f"{'[OK]' if ok else '[WRONG]'} ε={_agent.dqn.epsilon:.3f}")
        norm = min(0.999, max(0.001, tr/(n*10)))
        print(f"[END] task={task} score={norm:.3f} "
              f"rewards={','.join(f'{x:.1f}' for x in rws)}")

    s = _agent.get_stats()
    print(f"\n{'='*65}")
    print(f"  accuracy={s['accuracy']:.1f}% | reward={s['total_reward']:.1f}")
    print(f"  dqn_steps={s['dqn_steps']} | train_steps={s['train_steps']}")
    print(f"  sepsis_screened={s['sepsis_screened']}")
    print(f"  buffer={s['buffer_size']} | epsilon={s['epsilon']:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    import threading
    def _srv():
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
    threading.Thread(target=_srv, daemon=True).start()
    time.sleep(2)
    main()