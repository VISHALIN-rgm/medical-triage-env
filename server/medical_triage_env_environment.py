"""
Core Medical Triage Environment Logic - OpenEnv Compliant
"""

import random
import copy
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from pydantic import BaseModel

# ============================================================================
# Data Models
# ============================================================================

class PatientStatus(str, Enum):
    STABLE = "stable"
    URGENT = "urgent"
    CRITICAL = "critical"
    DISCHARGED = "discharged"
    DECEASED = "deceased"

class VitalSigns(BaseModel):
    heart_rate: int = 70
    blood_pressure_systolic: int = 120
    blood_pressure_diastolic: int = 80
    oxygen_saturation: float = 98.0
    temperature: float = 36.8
    respiratory_rate: int = 16

class Patient(BaseModel):
    id: str
    name: str = ""
    age: int = 0
    gender: str = "Unknown"
    symptoms: List[str] = []
    vitals: VitalSigns = VitalSigns()
    status: PatientStatus = PatientStatus.STABLE
    urgency_score: float = 0.5
    time_to_deterioration: int = 5
    chief_complaint: str = ""
    medical_history: List[str] = []
    processed: bool = False

class MedicalAction(BaseModel):
    type: str
    patient_id: str
    notes: Optional[str] = None

class MedicalObservation(BaseModel):
    patients: List[Patient]
    current_step: int
    max_steps: int
    task_id: str
    done: bool

class MedicalState(BaseModel):
    patients: List[Patient]
    step_count: int
    task_id: str
    actions_taken: List[MedicalAction]
    resource_usage: Dict[str, int]


# ============================================================================
# Main Environment Class
# ============================================================================

class MedicalTriageEnvironment:
    def __init__(self):
        self.patients = []
        self.step_count = 0
        self.max_steps = 20
        self.task_id = None
        self.actions_taken = []
        self.resource_usage = {"tests": 0, "treatments": 0}
        
    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> MedicalObservation:
        """Reset environment with new patients"""
        if seed is not None:
            random.seed(seed)
            
        self.task_id = task_id
        self.step_count = 0
        self.actions_taken = []
        self.resource_usage = {"tests": 0, "treatments": 0}
        
        # Set max steps based on task difficulty
        if task_id == "easy":
            self.max_steps = 3
            num_patients = 3
        elif task_id == "medium":
            self.max_steps = 5
            num_patients = 5
        else:
            self.max_steps = 8
            num_patients = 3
        
        # Generate patients
        self.patients = self._generate_patients(task_id, num_patients)
        
        return self._get_observation()
    
    def step(self, action: MedicalAction) -> Tuple[MedicalObservation, float, bool, dict]:
        """Execute an action"""
        self.step_count += 1
        self.actions_taken.append(action)
        
        reward = 0.0
        info = {"action_taken": action.type, "patient_id": action.patient_id}
        
        # Find patient
        patient = self._find_patient(action.patient_id)
        if not patient:
            return self._get_observation(), -1.0, True, {"error": "Patient not found"}
        
        # Convert action to lowercase
        action_type = action.type.lower()
        
        # Get expected action
        expected_action = self._get_expected_action(patient)
        info["expected_action"] = expected_action
        
        # Process action
        if action_type == "escalate":
            reward = self._handle_escalation(patient, expected_action)
        elif action_type == "discharge":
            reward = self._handle_discharge(patient, expected_action)
        elif action_type == "treat":
            reward = self._handle_treatment(patient, expected_action)
        elif action_type == "investigate":
            reward = self._handle_investigate(patient, expected_action)
        else:
            reward = -2.0
        
        # Mark as processed
        if action_type in ["escalate", "discharge", "treat"]:
            patient.processed = True
        
        # Step penalty
        reward -= 0.01
        
        # Update patients
        self._update_patients()
        
        # Check if done
        done = self._check_done()
        
        return self._get_observation(), reward, done, info
    
    def _get_expected_action(self, patient: Patient) -> str:
        urgency = patient.urgency_score
        if urgency > 0.7:
            return "ESCALATE"
        elif urgency > 0.3:
            return "TREAT"
        else:
            return "DISCHARGE"
    
    def _handle_escalation(self, patient: Patient, expected: str) -> float:
        if expected == "ESCALATE":
            patient.status = PatientStatus.DISCHARGED
            return 10.0
        elif expected == "TREAT" and patient.urgency_score > 0.5:
            return 3.0
        else:
            return -5.0
    
    def _handle_discharge(self, patient: Patient, expected: str) -> float:
        if expected == "DISCHARGE":
            patient.status = PatientStatus.DISCHARGED
            return 10.0
        elif expected == "TREAT" and patient.urgency_score < 0.4:
            return 2.0
        else:
            return -8.0
    
    def _handle_treatment(self, patient: Patient, expected: str) -> float:
        self.resource_usage["treatments"] += 1
        if expected == "TREAT":
            patient.urgency_score = max(0.0, patient.urgency_score - 0.2)
            patient.status = PatientStatus.STABLE
            return 10.0
        elif expected == "ESCALATE" and patient.urgency_score > 0.6:
            return 2.0
        else:
            return -3.0
    
    def _handle_investigate(self, patient: Patient, expected: str) -> float:
        self.resource_usage["tests"] += 1
        urgency = patient.urgency_score
        if 0.3 <= urgency <= 0.5:
            return 5.0
        elif expected == "TREAT" and self.resource_usage["tests"] <= 3:
            return 3.0
        else:
            return -2.0
    
    def _update_patients(self):
        for patient in self.patients:
            if patient.status not in [PatientStatus.DISCHARGED, PatientStatus.DECEASED]:
                patient.time_to_deterioration -= 1
                if patient.time_to_deterioration <= 0:
                    patient.urgency_score = min(1.0, patient.urgency_score + 0.2)
                    patient.time_to_deterioration = random.randint(2, 4)
    
    def _check_done(self) -> bool:
        all_processed = all(p.processed for p in self.patients)
        max_steps_reached = self.step_count >= self.max_steps
        return all_processed or max_steps_reached
    
    def _get_observation(self) -> MedicalObservation:
        return MedicalObservation(
            patients=copy.deepcopy(self.patients),
            current_step=self.step_count,
            max_steps=self.max_steps,
            task_id=self.task_id if self.task_id else "easy",
            done=self._check_done()
        )
    
    def state(self) -> MedicalState:
        return MedicalState(
            patients=copy.deepcopy(self.patients),
            step_count=self.step_count,
            task_id=self.task_id if self.task_id else "easy",
            actions_taken=self.actions_taken,
            resource_usage=self.resource_usage
        )
    
    def _find_patient(self, patient_id: str) -> Optional[Patient]:
        for p in self.patients:
            if p.id == patient_id:
                return p
        return None
    
    def _generate_patients(self, task_id: str, num_patients: int) -> List[Patient]:
        patients = []
        
        if task_id == "easy":
            scenarios = [
                {
                    "id": "P1", "name": "James Wilson", "age": 72,
                    "symptoms": ["chest pain", "shortness of breath"],
                    "vitals": VitalSigns(heart_rate=120, blood_pressure_systolic=85,
                                        oxygen_saturation=88, temperature=37.2),
                    "urgency_score": 0.85,
                    "chief_complaint": "Severe chest pain"
                },
                {
                    "id": "P2", "name": "Sarah Johnson", "age": 45,
                    "symptoms": ["mild headache"],
                    "vitals": VitalSigns(heart_rate=72, blood_pressure_systolic=118,
                                        oxygen_saturation=99, temperature=36.8),
                    "urgency_score": 0.15,
                    "chief_complaint": "Mild headache"
                },
                {
                    "id": "P3", "name": "Michael Brown", "age": 38,
                    "symptoms": ["runny nose"],
                    "vitals": VitalSigns(heart_rate=70, blood_pressure_systolic=120,
                                        oxygen_saturation=99, temperature=36.9),
                    "urgency_score": 0.10,
                    "chief_complaint": "Cold symptoms"
                }
            ]
            
            for scenario in scenarios:
                patient = Patient(
                    id=scenario["id"],
                    name=scenario["name"],
                    age=scenario["age"],
                    gender=random.choice(["Male", "Female"]),
                    symptoms=scenario["symptoms"],
                    vitals=scenario["vitals"],
                    status=self._get_status(scenario["urgency_score"]),
                    urgency_score=scenario["urgency_score"],
                    time_to_deterioration=5,
                    chief_complaint=scenario["chief_complaint"],
                    processed=False
                )
                patients.append(patient)
        
        return patients
    
    def _get_status(self, urgency: float) -> PatientStatus:
        if urgency > 0.7:
            return PatientStatus.CRITICAL
        elif urgency > 0.3:
            return PatientStatus.URGENT
        return PatientStatus.STABLE


# ============================================================================
# Main function
# ============================================================================

def main():
    print("Medical Triage Environment - Ready")
    print("Use server/app.py to start the API server")

if __name__ == "__main__":
    main()