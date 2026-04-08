"""
Pydantic models for Medical Triage Environment
Fixed: added to_feature_vector(), corrected MedicalAction types to match inference.py
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from enum import Enum
from datetime import datetime


class PatientStatus(str, Enum):
    CRITICAL = "critical"
    URGENT = "urgent"
    STABLE = "stable"
    DISCHARGED = "discharged"
    DECEASED = "deceased"


class VitalSigns(BaseModel):
    heart_rate: int = Field(ge=0, le=300, description="Beats per minute")
    blood_pressure_systolic: int = Field(ge=0, le=250, description="mmHg")
    blood_pressure_diastolic: int = Field(ge=0, le=150, description="mmHg")
    oxygen_saturation: float = Field(ge=0, le=100, description="Percentage")
    temperature: float = Field(ge=30, le=45, description="Celsius")
    respiratory_rate: int = Field(ge=0, le=60, description="Breaths per minute")

    def to_feature_vector(self) -> List[float]:
        """
        Returns a normalized feature vector for use in RL/ML models.
        All values scaled to roughly [0, 1] range.
        """
        return [
            self.heart_rate / 200.0,
            self.blood_pressure_systolic / 250.0,
            self.blood_pressure_diastolic / 150.0,
            self.oxygen_saturation / 100.0,
            (self.temperature - 30.0) / 15.0,
            self.respiratory_rate / 60.0,
        ]


class Patient(BaseModel):
    id: str
    name: str = "Unknown"
    age: int
    gender: str = "Unknown"
    symptoms: List[str]
    vitals: VitalSigns
    status: PatientStatus
    urgency_score: float = Field(ge=0.0, le=1.0)
    time_to_deterioration: int
    chief_complaint: str
    medical_history: List[str] = []
    arrival_time: datetime = Field(default_factory=datetime.now)

    def to_feature_vector(self) -> List[float]:
        """
        Full patient feature vector: vitals (6) + age (1) + urgency (1) = 8 features.
        Suitable as input to a DQN or any neural policy.
        """
        vitals_vec = self.vitals.to_feature_vector()
        return vitals_vec + [
            self.age / 100.0,
            self.urgency_score,
        ]


# FIX: action types now match what inference.py actually uses
class MedicalAction(BaseModel):
    type: Literal["discharge", "treat", "escalate", "investigate", "triage", "examine", "order_test"]
    patient_id: str
    test_name: Optional[str] = None
    treatment: Optional[str] = None
    notes: Optional[str] = None


class MedicalObservation(BaseModel):
    patients: List[Patient]
    available_tests: List[str] = ["CBC", "ECG", "CXR", "Troponin", "D-Dimer"]
    available_treatments: List[str] = ["oxygen", "fluids", "medication", "monitoring"]
    current_step: int
    max_steps: int
    task_id: str
    message: Optional[str] = None
    done: bool = False
    reward: Optional[float] = None


class MedicalState(BaseModel):
    patients: List[Patient]
    step_count: int
    task_id: str
    actions_taken: List[MedicalAction]
    resource_usage: Dict[str, int]