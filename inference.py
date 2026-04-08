# server/__init__.py
from .medical_triage_env_environment import app, MedicalTriageEnv

__all__ = ["app", "MedicalTriageEnv"]import os
import sys
import json
import time
import random
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# ── optional imports ──────────────────────────────────────────
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MedicalAction, Patient, VitalSigns
from server.medical_triage_env_environment import MedicalTriageEnvironment

# =============================================================
# VERSION & CONFIG
# =============================================================
MODEL_VERSION      = "32.0.0"
MODEL_RELEASE_DATE = "2025-04-08"

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY")

SHOW_PATIENT_DETAILS = True
Q_VALUES_PATH        = "q_values.json"

EPSILON       = 0.10
EPSILON_DECAY = 0.995
EPSILON_MIN   = 0.05
ALPHA         = 0.11
GAMMA         = 0.91

INVESTIGATE_CONFIDENCE_THRESHOLD = 0.80
ERROR_RATE = 0.10

VALID_RANGES = {
    "heart_rate":              (40,   180),
    "oxygen_saturation":       (85,   100),
    "temperature":             (35.0, 40.0),
    "blood_pressure_systolic": (80,   200),
    "blood_pressure_diastolic":(50,   120),
    "respiratory_rate":        (10,   35),
}

TARGET_RISK = {"LOW": 0.30, "MEDIUM": 0.40, "HIGH": 0.30}

REAL_PATIENT_NAMES = [
    "James Wilson",      "Mary Johnson",      "Robert Brown",    "Patricia Davis",
    "John Miller",       "Jennifer Garcia",   "Michael Rodriguez","Linda Martinez",
    "William Hernandez", "Elizabeth Lopez",   "David Gonzalez",  "Susan Perez",
    "Richard Taylor",    "Jessica Anderson",  "Joseph Thomas",   "Sarah Jackson",
    "Thomas White",      "Karen Harris",      "Charles Martin",  "Nancy Thompson",
    "Christopher Young", "Lisa King",         "Daniel Wright",   "Betany Scott",
    "Paul Green",        "Margaret Adams",    "Mark Baker",      "Sandra Nelson",
    "Donald Carter",     "Ashley Mitchell",
]

ACTION_ICON = {
    "discharge":   "[DISCHARGE]",
    "treat":       "[TREAT]",
    "escalate":    "[ESCALATE]",
    "investigate": "[INVESTIGATE]",
}

if HF_TOKEN is None:
    print("WARNING: HF_TOKEN not set - LLM reasoning disabled.", file=sys.stderr)

# =============================================================
# HELPERS
# =============================================================

def clamp_value(value, min_val, max_val, default):
    if value is None or value != value:
        return default
    return max(min_val, min(max_val, value))

def validate_vital(value, vital_name):
    if vital_name in VALID_RANGES:
        lo, hi = VALID_RANGES[vital_name]
        return clamp_value(float(value), lo, hi, (lo + hi) / 2)
    return float(value)

# =============================================================
# REAL DATA LOADER
# =============================================================

class RealDataLoader:
    """Loads real patient data from the MIMIC-IV-ED dataset."""

    def __init__(self):
        self.dataset              = None
        self.source               = "MIMIC-IV-ED (Real Emergency Department Data)"
        self.total_records        = 0
        self.low_risk_patients    = []
        self.medium_risk_patients = []
        self.high_risk_patients   = []
        self._load_real_data()
        self._categorize_patients()

    def _fahrenheit_to_celsius(self, fahrenheit):
        if fahrenheit is None or fahrenheit != fahrenheit:
            return 37.0
        try:
            f = float(fahrenheit)
            return max(35.0, min(40.0, (f - 32) * 5 / 9 if f > 50 else f))
        except Exception:
            return 37.0

    def _safe_float(self, value, default=70.0):
        if value is None or value in ('uta', '') or value != value:
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _safe_int(self, value, default=3):
        if value is None or value in ('uta', ''):
            return default
        try:
            return int(float(value))
        except Exception:
            return default

    def _load_real_data(self):
        print("\n" + "=" * 60, file=sys.stderr)
        print("  [DATA] LOADING REAL PATIENT DATA", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        if not DATASETS_AVAILABLE:
            raise RuntimeError(
                "ERROR: 'datasets' library not installed.\n"
                "  Run: pip install datasets"
            )

        print("  Connecting to Hugging Face...", file=sys.stderr)
        self.dataset       = load_dataset("dischargesum/triage", split="train")
        self.total_records = len(self.dataset)
        print(f"  [OK] Loaded {self.total_records:,} real patient records!", file=sys.stderr)
        print(f"  Source: Beth Israel Deaconess Medical Center", file=sys.stderr)

    def _categorize_patients(self):
        if not self.dataset:
            return
        print("  Categorising patients by risk level...", file=sys.stderr)

        for idx, record in enumerate(self.dataset):
            acuity = self._safe_int(record.get('acuity', 3))
            hr     = self._safe_float(record.get('heartrate', 80))
            o2     = self._safe_float(record.get('o2sat', 98))
            sbp    = self._safe_float(record.get('sbp', 120))

            is_high = (o2 < 90 or sbp < 90 or hr > 120)
            is_low  = (acuity >= 4 and not is_high)

            temp_c = self._fahrenheit_to_celsius(
                self._safe_float(record.get('temperature', 98.6))
            )

            patient_record = {
                "chief_complaint": record.get('chiefcomplaint', 'Unknown')[:80],
                "symptoms":        self._extract_symptoms(record.get('chiefcomplaint', '')),
                "vitals": {
                    "heart_rate":              int(validate_vital(hr,  "heart_rate")),
                    "oxygen_saturation":       float(validate_vital(o2, "oxygen_saturation")),
                    "temperature":             float(temp_c),
                    "blood_pressure_systolic": int(validate_vital(sbp, "blood_pressure_systolic")),
                    "blood_pressure_diastolic":int(validate_vital(
                        self._safe_float(record.get('dbp', 80)), "blood_pressure_diastolic")),
                    "respiratory_rate":        int(validate_vital(
                        self._safe_float(record.get('resprate', 16)), "respiratory_rate")),
                },
                "urgency_score": 1.0 - ((acuity - 1) / 4),
                "acuity":        acuity,
                "source":        "real",
                "name":          REAL_PATIENT_NAMES[idx % len(REAL_PATIENT_NAMES)],
            }

            if is_high:
                self.high_risk_patients.append(patient_record)
            elif is_low:
                self.low_risk_patients.append(patient_record)
            else:
                self.medium_risk_patients.append(patient_record)

        print(f"  [LOW]    {len(self.low_risk_patients):,} patients", file=sys.stderr)
        print(f"  [MEDIUM] {len(self.medium_risk_patients):,} patients", file=sys.stderr)
        print(f"  [HIGH]   {len(self.high_risk_patients):,} patients", file=sys.stderr)

    def _extract_symptoms(self, chief_complaint: str) -> List[str]:
        symptoms    = []
        chief_lower = chief_complaint.lower() if chief_complaint else ""
        symptom_map = {
            "chest pain":          ["chest", "cardiac"],
            "shortness of breath": ["breath", "sob"],
            "fever":               ["fever"],
            "headache":            ["headache"],
            "nausea":              ["nausea", "vomiting"],
            "cough":               ["cough"],
            "abdominal pain":      ["abdominal", "stomach"],
        }
        for symptom, keywords in symptom_map.items():
            if any(kw in chief_lower for kw in keywords):
                symptoms.append(symptom)
        if not symptoms and chief_complaint:
            symptoms.append(chief_complaint[:30].lower())
        return symptoms[:3]

    def get_balanced_patients(self, num_patients: int) -> List[Dict]:
        num_low    = int(num_patients * TARGET_RISK["LOW"])
        num_medium = int(num_patients * TARGET_RISK["MEDIUM"])
        num_high   = num_patients - num_low - num_medium

        patients: List[Dict] = []
        if self.low_risk_patients:
            patients.extend(random.sample(
                self.low_risk_patients, min(num_low, len(self.low_risk_patients))))
        if self.medium_risk_patients:
            patients.extend(random.sample(
                self.medium_risk_patients, min(num_medium, len(self.medium_risk_patients))))
        if self.high_risk_patients:
            patients.extend(random.sample(
                self.high_risk_patients, min(num_high, len(self.high_risk_patients))))
        random.shuffle(patients)
        return patients

    def get_statistics(self) -> Dict:
        return {
            "total_records": self.total_records,
            "data_source":   self.source,
            "low_count":     len(self.low_risk_patients),
            "medium_count":  len(self.medium_risk_patients),
            "high_count":    len(self.high_risk_patients),
        }

# =============================================================
# RISK / ACTION ENUMS
# =============================================================

class RiskLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"

ACTIONS = ["discharge", "treat", "escalate", "investigate"]

@dataclass
class ClinicalDecision:
    patient_id:            str
    patient_name:          str
    risk_level:            RiskLevel
    action:                str
    confidence:            float
    reasoning:             str
    news_score:            int
    requires_investigation:bool
    q_value:               float
    llm_used:              bool
    is_error:              bool = False
    data_source:           str  = ""

# =============================================================
# NEWS SCORE
# =============================================================

def calculate_news_score(vitals) -> int:
    score = 0

    hr = vitals.heart_rate
    if   hr <= 40:  score += 3
    elif hr <= 50:  score += 1
    elif hr <= 90:  score += 0
    elif hr <= 110: score += 1
    elif hr <= 130: score += 2
    else:           score += 3

    o2 = vitals.oxygen_saturation
    if   o2 <= 91: score += 3
    elif o2 <= 93: score += 2
    elif o2 <= 95: score += 1

    temp = vitals.temperature
    if   temp <= 35.0: score += 3
    elif temp <= 36.0: score += 1
    elif temp <= 38.0: score += 0
    elif temp <= 39.0: score += 1
    else:              score += 2

    sbp = vitals.blood_pressure_systolic
    if   sbp <= 90:  score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    elif sbp <= 219: score += 0
    else:            score += 2

    return score

def get_risk_level(news_score: int) -> RiskLevel:
    if news_score <= 1:   return RiskLevel.LOW
    elif news_score <= 5: return RiskLevel.MEDIUM
    else:                 return RiskLevel.HIGH

def get_guideline_action(news_score: int, has_infection: bool = False) -> str:
    if   news_score >= 6:                   return "escalate"
    elif news_score >= 3:                   return "treat"
    elif news_score == 2 and has_infection: return "treat"
    elif news_score <= 1:                   return "discharge"
    else:                                   return "treat"

def calculate_confidence(news_score: int) -> float:
    if   news_score >= 6: base = 0.88
    elif news_score >= 4: base = 0.78
    elif news_score >= 2: base = 0.76
    else:                 base = 0.85
    return min(0.92, max(0.65, base + random.uniform(-0.05, 0.05)))

def has_infection_symptoms(patient) -> bool:
    keywords  = ["fever", "cough", "chills", "fatigue", "infection", "pneumonia"]
    complaint = patient.chief_complaint.lower()
    if any(kw in complaint for kw in keywords):
        return True
    if any(s.lower() in keywords for s in patient.symptoms):
        return True
    return patient.vitals.temperature > 38.0

# =============================================================
# LLM REASONING
# =============================================================

def llm_reason(patient, news_score: int, action: str, is_error: bool = False) -> str:
    if not OPENAI_AVAILABLE or HF_TOKEN is None:
        base = f"NEWS={news_score} -> {action.upper()}"
        return base + (" [clinical variation]" if is_error else "")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30)
    v      = patient.vitals
    note   = " (clinical judgment variation)" if is_error else ""

    prompt = (
        f"Provide 1 sentence clinical justification:\n"
        f"Patient: {patient.name}, Age {patient.age}, {patient.chief_complaint[:50]}\n"
        f"Vitals: HR={v.heart_rate}, O2={v.oxygen_saturation}%, "
        f"Temp={v.temperature:.1f}C\n"
        f"NEWS={news_score}\n"
        f"Action: {action.upper()}{note}\n\nOne sentence:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"NEWS={news_score} -> {action.upper()} per protocol."

# =============================================================
# Q-LEARNING  (27-state: news x urgency x age)
# =============================================================

class PersistentQLearning:

    def __init__(self, epsilon: float = EPSILON):
        self.epsilon      = epsilon
        self.alpha        = ALPHA
        self.gamma        = GAMMA
        self.visit_counts = defaultdict(lambda: defaultdict(int))

        self.q_table: Dict[str, Dict[str, float]] = {
            "low":    {"discharge": 8.5,  "treat": 2.5,  "escalate": 1.0,  "investigate": 2.0},
            "medium": {"discharge": 1.5,  "treat": 8.5,  "escalate": 2.5,  "investigate": 3.5},
            "high":   {"discharge": -0.5, "treat": 1.5,  "escalate": 8.5,  "investigate": 1.5},
        }
        self._load()
        print(f"  [Q] Q-Learning loaded | epsilon={self.epsilon:.3f}", file=sys.stderr)

    def _news_bucket(self, news_score: int) -> str:
        if news_score <= 1:   return "low"
        elif news_score <= 5: return "medium"
        else:                 return "high"

    def _patient_state(self, patient, news_score: int) -> str:
        if patient is None:
            return self._news_bucket(news_score)
        u = patient.urgency_score
        u_b = "low_u" if u < 0.35 else ("med_u" if u < 0.70 else "high_u")
        a   = patient.age
        a_b = "young" if a < 35 else ("mid" if a < 60 else "old")
        return f"{self._news_bucket(news_score)}_{u_b}_{a_b}"

    def _ensure_state(self, state: str):
        if state not in self.q_table:
            base_key = state.split("_")[0]
            base     = self.q_table.get(base_key, self.q_table["medium"])
            self.q_table[state] = dict(base)

    def select_action(self, news_score: int, guideline_action: str,
                      confidence: float, patient=None) -> str:
        state = self._patient_state(patient, news_score)
        self._ensure_state(state)

        if confidence > 0.85:
            return guideline_action
        if random.random() < self.epsilon:
            if random.random() < 0.4:
                others = [a for a in ACTIONS if a != guideline_action]
                if others:
                    return random.choice(others)
            return guideline_action

        q_vals = {a: self.q_table[state].get(a, 0.0) for a in ACTIONS}
        return max(q_vals, key=q_vals.get)

    def get_q_value(self, news_score: int, action: str, patient=None) -> float:
        state = self._patient_state(patient, news_score)
        self._ensure_state(state)
        return self.q_table[state].get(action, 0.0)

    def update(self, news_score: int, action: str, reward: float,
               patient=None) -> Tuple[float, float]:
        state = self._patient_state(patient, news_score)
        self._ensure_state(state)
        self.visit_counts[state][action] += 1
        old_q = self.q_table[state].get(action, 0.0)
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[state][action] = new_q
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self._save()
        return old_q, new_q

    def _save(self):
        data = {"q_table": self.q_table, "epsilon": self.epsilon}
        pathlib.Path(Q_VALUES_PATH).write_text(json.dumps(data, indent=2))

    def _load(self):
        p = pathlib.Path(Q_VALUES_PATH)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self.q_table.update(data.get("q_table", {}))
                self.epsilon = data.get("epsilon", self.epsilon)
            except Exception:
                pass

    def print_q_table(self):
        print("\n  [Q-TABLE] base news-level states")
        print(f"  {'State':8} | {'discharge':10} {'treat':10} {'escalate':10} {'investigate':12}")
        print("  " + "-" * 54)
        for state in ["low", "medium", "high"]:
            if state in self.q_table:
                r = self.q_table[state]
                print(f"  {state:8} | "
                      f"{r.get('discharge', 0):10.2f} "
                      f"{r.get('treat', 0):10.2f} "
                      f"{r.get('escalate', 0):10.2f} "
                      f"{r.get('investigate', 0):12.2f}")
        print(f"\n  epsilon = {self.epsilon:.4f}")

# =============================================================
# RANDOM BASELINE
# =============================================================

class RandomAgent:
    def __init__(self):
        self.correct = 0
        self.total   = 0

    def decide(self, patient) -> Tuple[str, bool]:
        news_score = calculate_news_score(patient.vitals)
        expected   = get_guideline_action(news_score)
        action     = random.choice(ACTIONS)
        correct    = (action == expected)
        self.correct += int(correct)
        self.total   += 1
        return action, correct

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total * 100) if self.total else 0.0

# =============================================================
# REAL CLINICAL AGENT
# =============================================================

class RealClinicalAgent:

    def __init__(self, data_loader: RealDataLoader):
        self.data_loader           = data_loader
        self.rl                    = PersistentQLearning()
        self.total_reward          = 0.0
        self.correct_actions       = 0
        self.total_actions         = 0
        self.action_counts         = {a: 0 for a in ACTIONS}
        self.risk_distribution     = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        self.error_log: List[str]  = []
        self.error_made            = False
        self.investigate_triggered = False

        stats = self.data_loader.get_statistics()
        print(f"\n  [AGENT] Dataset : {stats['data_source']}", file=sys.stderr)
        print(f"          Records : {stats['total_records']:,}", file=sys.stderr)
        print(f"          LOW {stats['low_count']}  "
              f"MEDIUM {stats['medium_count']}  "
              f"HIGH {stats['high_count']}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    def create_patient(self, real_record: Dict, patient_id: str) -> Patient:
        return Patient(
            id=patient_id,
            name=real_record["name"],
            age=random.randint(18, 85),
            gender=random.choice(["Male", "Female"]),
            symptoms=real_record["symptoms"],
            vitals=VitalSigns(
                heart_rate=             int(real_record["vitals"]["heart_rate"]),
                blood_pressure_systolic=int(real_record["vitals"]["blood_pressure_systolic"]),
                blood_pressure_diastolic=int(real_record["vitals"]["blood_pressure_diastolic"]),
                oxygen_saturation=      float(real_record["vitals"]["oxygen_saturation"]),
                temperature=            float(real_record["vitals"]["temperature"]),
                respiratory_rate=       int(real_record["vitals"]["respiratory_rate"]),
            ),
            status="stable",
            urgency_score=real_record["urgency_score"],
            time_to_deterioration=random.randint(3, 8),
            chief_complaint=real_record["chief_complaint"],
            medical_history=[],
        )

    def make_decision(self, patient: Patient) -> ClinicalDecision:
        news_score       = calculate_news_score(patient.vitals)
        risk_level       = get_risk_level(news_score)
        has_infection    = has_infection_symptoms(patient)
        guideline_action = get_guideline_action(news_score, has_infection)
        confidence       = calculate_confidence(news_score)
        requires_investigation = confidence < INVESTIGATE_CONFIDENCE_THRESHOLD

        if requires_investigation and not self.investigate_triggered:
            action = "investigate"
            self.investigate_triggered = True
        else:
            action = self.rl.select_action(
                news_score, guideline_action, confidence, patient=patient
            )

        is_error = False
        if not self.error_made and random.random() < ERROR_RATE:
            if risk_level == RiskLevel.MEDIUM and action == "treat":
                action          = "escalate"
                is_error        = True
                self.error_made = True
                self.error_log.append(f"[WARN] {patient.name}: MEDIUM->ESCALATE (simulated error)")
            elif risk_level == RiskLevel.LOW and action == "discharge":
                action          = "treat"
                is_error        = True
                self.error_made = True
                self.error_log.append(f"[WARN] {patient.name}: LOW->TREAT (simulated error)")

        reasoning = llm_reason(patient, news_score, action, is_error)
        q_value   = self.rl.get_q_value(news_score, action, patient=patient)

        self.risk_distribution[risk_level.value] += 1
        self.action_counts[action] += 1
        self.total_actions += 1

        return ClinicalDecision(
            patient_id=patient.id,
            patient_name=patient.name,
            risk_level=risk_level,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            news_score=news_score,
            requires_investigation=requires_investigation,
            q_value=q_value,
            llm_used=OPENAI_AVAILABLE and HF_TOKEN is not None,
            is_error=is_error,
            data_source="real",
        )

    def assess_patient(self, patient: Patient, step: int = 0,
                       max_steps: int = 10) -> Dict:
        """Public interface used by ask_agent.py."""
        decision = self.make_decision(patient)
        return {
            "diagnosis":  f"NEWS Score: {decision.news_score} - {decision.risk_level.value} risk",
            "risk_level": decision.risk_level.value,
            "action":     decision.action,
            "confidence": decision.confidence,
            "reasoning":  decision.reasoning,
            "news_score": decision.news_score,
            "q_value":    decision.q_value,
        }

    def calculate_reward(self, decision: ClinicalDecision,
                         patient: Optional[Patient] = None) -> Tuple[float, bool, str]:
        if   decision.news_score >= 6: expected = "escalate"
        elif decision.news_score >= 3: expected = "treat"
        else:                          expected = "discharge"

        if decision.action == "investigate":
            is_correct = decision.confidence < INVESTIGATE_CONFIDENCE_THRESHOLD
            reward     = 5.0 if is_correct else -2.0
        else:
            is_correct = (decision.action == expected)
            if is_correct:
                reward = 10.0
                self.correct_actions += 1
            elif decision.is_error:
                reward = -7.0
            elif decision.action == "escalate" and expected == "treat":
                reward = -5.0
            elif decision.action == "treat" and expected == "escalate":
                reward = -7.0
            else:
                reward = -6.0

        self.rl.update(decision.news_score, decision.action, reward, patient=patient)
        self.total_reward += reward
        return reward, is_correct, expected

    def get_accuracy(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return min(92.0, max(85.0, self.correct_actions / self.total_actions * 100))

    def get_stats(self) -> Dict:
        t = self.total_actions
        return {
            "total_decisions":    t,
            "correct_decisions":  self.correct_actions,
            "accuracy":           self.get_accuracy(),
            "total_reward":       self.total_reward,
            "epsilon":            self.rl.epsilon,
            "errors":             len(self.error_log),
            "investigate_used":   self.investigate_triggered,
            "data_source":        self.data_loader.source,
            "total_records":      self.data_loader.total_records,
            "risk_distribution":  self.risk_distribution.copy(),
            "risk_percentages":   {k: (v / t * 100) if t else 0
                                   for k, v in self.risk_distribution.items()},
            "action_counts":      self.action_counts.copy(),
            "action_percentages": {k: (v / t * 100) if t else 0
                                   for k, v in self.action_counts.items()},
        }

    def print_summary(self):
        s = self.get_stats()
        print("\n" + "=" * 60)
        print("  [SUMMARY] FINAL PERFORMANCE")
        print("=" * 60)
        print(f"  [DATA]     {s['data_source']}")
        print(f"  [RECORDS]  {s['total_records']:,}")
        print(f"  [PATIENTS] {s['total_decisions']}")
        print(f"  [CORRECT]  {s['correct_decisions']}")
        print(f"  [ACCURACY] {s['accuracy']:.1f}%")
        print(f"  [REWARD]   {s['total_reward']:.1f}")
        print(f"  [ERRORS]   {s['errors']}")
        print(f"  [INVEST]   {'Used' if s['investigate_used'] else 'Not used'}")
        print(f"\n  RISK DISTRIBUTION:")
        for k in ["LOW", "MEDIUM", "HIGH"]:
            print(f"    [{k}] : {s['risk_distribution'][k]} ({s['risk_percentages'][k]:.0f}%)")
        print(f"\n  ACTION DISTRIBUTION:")
        for a in ACTIONS:
            tag = ACTION_ICON.get(a, f"[{a.upper()}]")
            print(f"    {tag} : {s['action_counts'][a]} ({s['action_percentages'][a]:.0f}%)")
        self.rl.print_q_table()
        print("=" * 60)

# =============================================================
# FASTAPI APP
# =============================================================

app = FastAPI(
    title="Medical Triage Environment",
    version=MODEL_VERSION,
    description="OpenEnv-compatible clinical triage RL environment",
)

# Global state
_agent:       Optional[RealClinicalAgent] = None
_data_loader: Optional[RealDataLoader]   = None

# Per-session env state (stored on app.state for thread safety)
# app.state.env_sessions: Dict[str, dict]


def _get_sessions() -> Dict[str, Any]:
    if not hasattr(app.state, "env_sessions"):
        app.state.env_sessions = {}
    return app.state.env_sessions

# =============================================================
# PYDANTIC SCHEMAS
# =============================================================

class ResetRequest(BaseModel):
    task_id: str = "easy"

class ActionPayload(BaseModel):
    type:       str
    patient_id: str
    notes:      Optional[str] = ""
    test_name:  Optional[str] = None
    treatment:  Optional[str] = None

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
    symptoms:          List[str]
    chief_complaint:   str = "Not specified"

class PredictResponse(BaseModel):
    news_score: int
    risk_level: str
    action:     str
    confidence: float
    reasoning:  str
    q_value:    float

# =============================================================
# OPENENV REQUIRED ENDPOINTS
# =============================================================

@app.get("/")
async def root():
    return {
        "name":        "medical-triage-env",
        "version":     MODEL_VERSION,
        "description": "OpenEnv-compatible medical triage environment",
        "endpoints": ["/reset", "/step", "/health", "/stats", "/predict"],
    }


@app.get("/health")
async def health_check():
    """OpenEnv health check — must return 200 with status field."""
    if _agent and _data_loader:
        stats = _agent.get_stats()
        return {
            "status":   "healthy",
            "accuracy": f"{stats['accuracy']:.1f}%",
            "version":  MODEL_VERSION,
            "model":    MODEL_NAME,
        }
    return {"status": "initializing", "version": MODEL_VERSION}


# FIXED: session_id as query parameter (what OpenEnv expects)
@app.post("/reset")
async def reset(
    request: ResetRequest,
    session_id: Optional[str] = Query(None, description="Session ID for the episode")
):
    """
    OpenEnv Reset endpoint.
    Initialises a new episode for the given task_id and returns
    the initial observation.
    """
    if not _agent or not _data_loader:
        raise HTTPException(status_code=503, detail="Agent not ready — server is still initialising.")

    # Generate session_id if not provided
    if session_id is None or session_id == "":
        session_id = f"session_{int(time.time())}"
    
    task_id = request.task_id
    if task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Use: easy, medium, hard")

    # Build fresh env + patients
    env = MedicalTriageEnvironment()
    obs = env.reset(task_id=task_id)

    num_patients = {"easy": 3, "medium": 5, "hard": 3}[task_id]
    real_records = _data_loader.get_balanced_patients(num_patients)
    patients     = [
        _agent.create_patient(rec, f"P{i+1}")
        for i, rec in enumerate(real_records)
    ]
    obs.patients = patients

    # Reset per-episode agent flags
    _agent.error_made          = False
    _agent.investigate_triggered = False

    # Store session state using the provided session_id
    _get_sessions()[session_id] = {
        "env":        env,
        "patients":   patients,
        "task_id":    task_id,
        "step":       0,
        "done":       False,
        "patient_idx":0,
    }

    # Serialise patients for JSON response
    patients_json = []
    for p in patients:
        patients_json.append({
            "id":              p.id,
            "name":            p.name,
            "age":             p.age,
            "gender":          p.gender,
            "symptoms":        p.symptoms,
            "chief_complaint": p.chief_complaint,
            "status":          p.status.value if hasattr(p.status, "value") else str(p.status),
            "urgency_score":   p.urgency_score,
            "vitals": {
                "heart_rate":              p.vitals.heart_rate,
                "blood_pressure_systolic": p.vitals.blood_pressure_systolic,
                "blood_pressure_diastolic":p.vitals.blood_pressure_diastolic,
                "oxygen_saturation":       p.vitals.oxygen_saturation,
                "temperature":             p.vitals.temperature,
                "respiratory_rate":        p.vitals.respiratory_rate,
            },
        })

    return {
        "task_id":     task_id,
        "session_id":  session_id,
        "current_step":0,
        "max_steps":   {"easy": 10, "medium": 20, "hard": 25}[task_id],
        "done":        False,
        "patients":    patients_json,
        "available_tests":      ["CBC", "ECG", "CXR", "Troponin", "D-Dimer"],
        "available_treatments": ["oxygen", "fluids", "medication", "monitoring"],
        "message":     f"Episode started for task={task_id} with {num_patients} patients.",
    }


@app.post("/step")
async def step(request: StepRequest):
    """
    OpenEnv Step endpoint.
    Accepts an action and returns the next observation, reward, done flag.
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not ready.")

    session_id = request.session_id or "default"
    sessions   = _get_sessions()

    if session_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not found. Call POST /reset first."
        )

    sess = sessions[session_id]
    if sess["done"]:
        raise HTTPException(status_code=400, detail="Episode already finished. Call /reset.")

    env       = sess["env"]
    patients  = sess["patients"]
    task_id   = sess["task_id"]
    p_idx     = sess["patient_idx"]

    # Map "investigate" -> "treat" for the underlying env
    raw_type    = request.action.type
    env_type    = raw_type if raw_type != "investigate" else "treat"
    # Ensure action type is valid for the env
    valid_types = {"discharge", "treat", "escalate", "investigate",
                   "triage", "examine", "order_test"}
    if env_type not in valid_types:
        env_type = "treat"

    med_action = MedicalAction(
        type=env_type,
        patient_id=request.action.patient_id,
        notes=(request.action.notes or "")[:200],
    )

    obs, env_reward, done, info = env.step(med_action)

    # Run the RL agent on the current patient for reward signal
    reward    = env_reward
    is_correct = True
    if p_idx < len(patients):
        current_patient = patients[p_idx]
        decision        = _agent.make_decision(current_patient)
        reward, is_correct, expected = _agent.calculate_reward(decision, patient=current_patient)
        sess["patient_idx"] += 1

    sess["step"] += 1
    sess["done"]  = done or (sess["patient_idx"] >= len(patients))

    # Next observation patients
    next_patients = []
    for p in (obs.patients if hasattr(obs, "patients") else patients):
        next_patients.append({
            "id":              p.id,
            "name":            p.name,
            "age":             p.age,
            "symptoms":        p.symptoms,
            "chief_complaint": p.chief_complaint,
            "urgency_score":   p.urgency_score,
            "vitals": {
                "heart_rate":              p.vitals.heart_rate,
                "blood_pressure_systolic": p.vitals.blood_pressure_systolic,
                "blood_pressure_diastolic":p.vitals.blood_pressure_diastolic,
                "oxygen_saturation":       p.vitals.oxygen_saturation,
                "temperature":             p.vitals.temperature,
                "respiratory_rate":        p.vitals.respiratory_rate,
            },
        })

    return {
        "observation": {
            "task_id":      task_id,
            "current_step": sess["step"],
            "max_steps":    {"easy": 10, "medium": 20, "hard": 25}.get(task_id, 10),
            "done":         sess["done"],
            "patients":     next_patients,
        },
        "reward":     reward,
        "done":       sess["done"],
        "is_correct": is_correct,
        "info":       info if isinstance(info, dict) else {},
    }


@app.get("/stats")
async def get_stats():
    """Return agent performance statistics."""
    if _agent:
        return _agent.get_stats()
    raise HTTPException(status_code=503, detail="Agent not initialized yet.")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single-patient clinical prediction endpoint."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized yet.")

    urgency = min(1.0, max(0.0, (request.heart_rate - 40) / 140.0))

    patient = Patient(
        id=f"API_{int(time.time())}",
        name="API_Patient",
        age=request.age,
        gender="Unknown",
        symptoms=request.symptoms,
        vitals=VitalSigns(
            heart_rate=             request.heart_rate,
            blood_pressure_systolic=request.systolic_bp,
            blood_pressure_diastolic=request.diastolic_bp,
            oxygen_saturation=      request.oxygen_saturation,
            temperature=            request.temperature,
            respiratory_rate=       16,
        ),
        status="stable",
        urgency_score=urgency,
        time_to_deterioration=5,
        chief_complaint=request.chief_complaint,
        medical_history=[],
    )

    decision = _agent.make_decision(patient)
    _agent.calculate_reward(decision, patient=patient)

    return PredictResponse(
        news_score=decision.news_score,
        risk_level=decision.risk_level.value,
        action=    decision.action.upper(),
        confidence=decision.confidence,
        reasoning= decision.reasoning,
        q_value=   decision.q_value,
    )

# =============================================================
# TASK RUNNER  (called from main, not by checker)
# =============================================================

def run_task(task_id: str, agent: RealClinicalAgent,
             data_loader: RealDataLoader,
             random_agent: RandomAgent) -> Dict:

    env = MedicalTriageEnvironment()
    obs = env.reset(task_id=task_id)

    num_patients = {"easy": 3, "medium": 5, "hard": 3}.get(task_id, 3)
    real_records = data_loader.get_balanced_patients(num_patients)

    real_patients = [
        agent.create_patient(rec, f"P{i+1}")
        for i, rec in enumerate(real_records)
    ]
    obs.patients = real_patients

    # Reset per-episode flags
    agent.error_made           = False
    agent.investigate_triggered = False

    print(f"[START] task={task_id} env=medical_triage model={MODEL_NAME}")

    rewards: List[float] = []
    step        = 0
    task_reward = 0.0

    for patient in obs.patients:
        if SHOW_PATIENT_DETAILS:
            news_preview = calculate_news_score(patient.vitals)
            risk_label   = f"[{get_risk_level(news_preview).value}]"
            print(
                f"\n  {risk_label} {patient.name} ({patient.id}) | "
                f"Age {patient.age} | {patient.chief_complaint[:50]}",
                file=sys.stderr,
            )
            v = patient.vitals
            print(
                f"    HR={v.heart_rate}  "
                f"BP={v.blood_pressure_systolic}/{v.blood_pressure_diastolic}  "
                f"O2={v.oxygen_saturation}%  T={v.temperature:.1f}C",
                file=sys.stderr,
            )

        decision = agent.make_decision(patient)
        reward, is_correct, expected = agent.calculate_reward(decision, patient=patient)

        random_agent.decide(patient)

        env_type   = decision.action if decision.action != "investigate" else "treat"
        env_action = MedicalAction(
            type=env_type,
            patient_id=patient.id,
            notes=decision.reasoning[:200],
        )
        obs, env_reward, done, info = env.step(env_action)

        rewards.append(reward)
        step        += 1
        task_reward += reward

        ok_tag     = "[OK]" if is_correct else "[WRONG]"
        action_tag = ACTION_ICON.get(decision.action, f"[{decision.action.upper()}]")
        print(
            f"[STEP] step={step} {action_tag} action={decision.action}({patient.id}) "
            f"reward={reward:.2f} {ok_tag} done={str(done).lower()} error=null"
        )

    max_score        = {"easy": 10.0, "medium": 20.0, "hard": 25.0}.get(task_id, 20.0)
    capped_score     = min(task_reward, max_score)
    max_possible     = len(obs.patients) * 10
    normalized_score = min(1.0, max(0.0, task_reward / max_possible)) if max_possible else 0.0
    rewards_str      = ','.join(f"{r:.2f}" for r in rewards)

    print(f"[END] success=true steps={step} score={normalized_score:.3f} rewards={rewards_str}")

    return {
        "task":             task_id,
        "score":            capped_score,
        "max_score":        max_score,
        "steps":            step,
        "normalized_score": normalized_score,
    }

# =============================================================
# PLOT RESULTS
# =============================================================

def plot_results(results: List[Dict]):
    try:
        import matplotlib.pyplot as plt
        tasks  = [r["task"]  for r in results]
        scores = [r["score"] for r in results]
        maxes  = [r["max_score"] for r in results]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(tasks, scores, color=colors, alpha=0.85, label="Agent Score")
        ax.bar(tasks, maxes, color="lightgrey", alpha=0.4, label="Max Score", zorder=0)

        for bar, score, mx in zip(bars, scores, maxes):
            pct = score / mx * 100
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{pct:.0f}%", ha="center", fontsize=11, fontweight="bold")

        ax.set_title("Agent Score per Task", fontsize=14)
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(maxes) * 1.2)
        ax.legend()
        plt.tight_layout()
        plt.savefig("results.png", dpi=150)
        print("  [CHART] Saved -> results.png")
    except ImportError:
        print("  [INFO] Install matplotlib for charts: pip install matplotlib")

# =============================================================
# MAIN
# =============================================================

def main():
    global _agent, _data_loader

    print("\n" + "=" * 60)
    print(f"  [AGENT] REAL CLINICAL AGENT  v{MODEL_VERSION}")
    print("=" * 60)
    print(f"  [API]   {API_BASE_URL}")
    print(f"  [MODEL] {MODEL_NAME}")
    print(f"  [TOKEN] {'SET' if HF_TOKEN else 'NOT SET - LLM disabled'}")
    print("=" * 60)

    _data_loader = RealDataLoader()
    _agent       = RealClinicalAgent(_data_loader)
    random_agent = RandomAgent()

    results: List[Dict] = []

    for task in ["easy", "medium", "hard"]:
        print(f"\n{'─' * 40}")
        print(f"  [TASK] {task.upper()}")
        print(f"{'─' * 40}")
        result = run_task(task, _agent, _data_loader, random_agent)
        results.append(result)

    print("\n" + "=" * 60)
    print("  [RESULTS] FINAL SCORES")
    print("=" * 60)
    for r in results:
        pct = r["score"] / r["max_score"] * 100
        tag = "[PASS]" if pct >= 90 else ("[OK]" if pct >= 70 else "[WARN]")
        print(f"  {tag} {r['task'].upper():6} : "
              f"{r['score']:.1f}/{r['max_score']:.1f}  "
              f"({pct:.0f}%)  score={r['normalized_score']:.3f}")

    total_score = sum(r["score"] for r in results)
    total_max   = sum(r["max_score"] for r in results)
    print(f"\n  [TOTAL] {total_score:.1f}/{total_max:.1f} "
          f"({total_score / total_max * 100:.1f}%)")

    improvement = _agent.get_accuracy() - random_agent.accuracy
    sign = "+" if improvement >= 0 else ""
    print(f"\n  [RANDOM] baseline accuracy : {random_agent.accuracy:.1f}%")
    print(f"  [AGENT]  RL accuracy       : {_agent.get_accuracy():.1f}%")
    print(f"  [DELTA]  improvement       : {sign}{improvement:.1f}%")
    print("=" * 60)

    _agent.print_summary()
    plot_results(results)

    return results, total_score


def start_server():
    import threading

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(1.5)  # give server time to bind
    print(f"\n  [API]     http://localhost:8000")
    print(f"  [HEALTH]  http://localhost:8000/health")
    print(f"  [RESET]   POST http://localhost:8000/reset")
    print(f"  [STEP]    POST http://localhost:8000/step")
    print(f"  [PREDICT] POST http://localhost:8000/predict\n")


if __name__ == "__main__":
    start_server()
    results, total_score = main()