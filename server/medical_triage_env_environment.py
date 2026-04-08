"""
Core Medical Triage Environment Logic
"""

import random
import copy
from typing import List, Dict, Tuple, Optional
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_data_loader import RealMedicalDataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Patient, VitalSigns, PatientStatus, MedicalAction, MedicalObservation, MedicalState

class MedicalTriageEnvironment:
    def __init__(self):
        self.patients = []
        self.step_count = 0
        self.max_steps = 20
        self.task_id = None
        self.actions_taken = []
        self.resource_usage = {"tests": 0, "treatments": 0}
        
    def reset(self, task_id: str = "easy", num_patients: int = 3):
        """Reset environment with new patients"""
        self.task_id = task_id
        self.step_count = 0
        self.actions_taken = []
        self.resource_usage = {"tests": 0, "treatments": 0}
        
        # Set max steps based on task difficulty
        if task_id == "easy":
            self.max_steps = 10
            num_patients = 3
        elif task_id == "medium":
            self.max_steps = 20
            num_patients = 5
        else:  # hard
            self.max_steps = 25
            num_patients = 3
        
        # Generate patients
        self.patients = []
        names = ["John Smith", "Mary Johnson", "Robert Williams", "Patricia Brown", "Michael Jones"]
        
        for i in range(num_patients):
            urgency = random.random()
            
            # Adjust urgency based on task
            if task_id == "easy":
                urgency = 0.8 if i == 0 else 0.3
            elif task_id == "medium":
                urgency = random.uniform(0.2, 0.9)
            else:  # hard
                urgency = random.uniform(0.4, 0.8)
            
            patient = Patient(
                id=f"P{i+1}",
                name=names[i % len(names)],
                age=random.randint(20, 85),
                gender=random.choice(["Male", "Female"]),
                symptoms=self._get_symptoms(urgency),
                vitals=self._get_vitals(urgency),
                status=self._get_status(urgency),
                urgency_score=urgency,
                time_to_deterioration=random.randint(2, 5),
                chief_complaint=self._get_chief_complaint(urgency),
                medical_history=self._get_medical_history(urgency)
            )
            self.patients.append(patient)
        
        return self._get_observation()
    
    def step(self, action: MedicalAction) -> Tuple[MedicalObservation, float, bool, dict]:
        """Execute an action"""
        self.step_count += 1
        self.actions_taken.append(action)
        
        reward = 0.0
        info = {}
        
        # Process action
        if action.type == "escalate":
            reward = self._handle_escalation(action)
        elif action.type == "discharge":
            reward = self._handle_discharge(action)
        elif action.type == "triage":
            reward = self._handle_triage(action)
        elif action.type == "order_test":
            reward = self._handle_test(action)
        elif action.type == "treat":
            reward = self._handle_treatment(action)
        elif action.type == "examine":
            reward = 0.05
        
        # Step penalty to encourage efficiency
        reward -= 0.01
        
        # Update patient deterioration
        self._update_patients()
        
        # Check if done
        done = self._check_done()
        
        observation = self._get_observation()
        return observation, reward, done, info
    
    def state(self) -> MedicalState:
        """Return current state"""
        return MedicalState(
            patients=copy.deepcopy(self.patients),
            step_count=self.step_count,
            task_id=self.task_id,
            actions_taken=self.actions_taken,
            resource_usage=self.resource_usage
        )
    
    # ============== Reward Handlers ==============
    
    def _handle_escalation(self, action: MedicalAction) -> float:
        patient = self._find_patient(action.patient_id)
        if not patient:
            return -0.1
        
        if patient.urgency_score > 0.7:
            patient.status = PatientStatus.DISCHARGED
            return 1.0
        elif patient.urgency_score > 0.5:
            return 0.3
        else:
            return -0.2
    
    def _handle_discharge(self, action: MedicalAction) -> float:
        patient = self._find_patient(action.patient_id)
        if not patient:
            return -0.1
        
        if patient.urgency_score < 0.3:
            patient.status = PatientStatus.DISCHARGED
            return 0.5
        elif patient.urgency_score < 0.5:
            return 0.2
        else:
            return -0.3
    
    def _handle_triage(self, action: MedicalAction) -> float:
        patient = self._find_patient(action.patient_id)
        if not patient:
            return -0.05
        
        notes = str(action.notes).lower() if action.notes else ""
        if "critical" in notes and patient.urgency_score > 0.7:
            return 0.3
        elif "urgent" in notes and patient.urgency_score > 0.3:
            return 0.2
        elif "stable" in notes and patient.urgency_score < 0.3:
            return 0.2
        return 0.1
    
    def _handle_test(self, action: MedicalAction) -> float:
        self.resource_usage["tests"] += 1
        if self.resource_usage["tests"] > 5:
            return -0.1
        return 0.1
    
    def _handle_treatment(self, action: MedicalAction) -> float:
        self.resource_usage["treatments"] += 1
        patient = self._find_patient(action.patient_id)
        if patient and patient.urgency_score > 0.5:
            # Treatment improves patient
            patient.urgency_score = max(0.0, patient.urgency_score - 0.2)
            return 0.4
        return 0.1
    
    # ============== Helper Methods ==============
    
    def _update_patients(self):
        for patient in self.patients:
            if patient.status not in [PatientStatus.DISCHARGED, PatientStatus.DECEASED]:
                patient.time_to_deterioration -= 1
                if patient.time_to_deterioration <= 0:
                    patient.urgency_score = min(1.0, patient.urgency_score + 0.2)
                    patient.time_to_deterioration = random.randint(2, 4)
                    
                    if patient.urgency_score > 0.9:
                        patient.status = PatientStatus.CRITICAL
    
    def _check_done(self) -> bool:
        all_done = all(p.status in [PatientStatus.DISCHARGED, PatientStatus.DECEASED] 
                      for p in self.patients)
        max_steps_reached = self.step_count >= self.max_steps
        return all_done or max_steps_reached
    
    def _get_observation(self) -> MedicalObservation:
        return MedicalObservation(
            patients=copy.deepcopy(self.patients),
            current_step=self.step_count,
            max_steps=self.max_steps,
            task_id=self.task_id,
            done=self._check_done()
        )
    
    def _find_patient(self, patient_id: str) -> Optional[Patient]:
        for p in self.patients:
            if p.id == patient_id:
                return p
        return None
    
    def _get_symptoms(self, urgency: float) -> List[str]:
        if urgency > 0.7:
            return ["chest pain", "shortness of breath", "dizziness", "nausea"]
        elif urgency > 0.3:
            return ["fever", "cough", "fatigue", "headache"]
        else:
            return ["mild pain", "runny nose", "minor scratch"]
    
    def _get_vitals(self, urgency: float) -> VitalSigns:
        if urgency > 0.7:
            return VitalSigns(
                heart_rate=120, blood_pressure_systolic=80, blood_pressure_diastolic=50,
                oxygen_saturation=88, temperature=39.0, respiratory_rate=28
            )
        elif urgency > 0.3:
            return VitalSigns(
                heart_rate=95, blood_pressure_systolic=110, blood_pressure_diastolic=70,
                oxygen_saturation=96, temperature=38.0, respiratory_rate=18
            )
        else:
            return VitalSigns(
                heart_rate=70, blood_pressure_systolic=120, blood_pressure_diastolic=80,
                oxygen_saturation=99, temperature=36.8, respiratory_rate=14
            )
    
    def _get_status(self, urgency: float) -> PatientStatus:
        if urgency > 0.7:
            return PatientStatus.CRITICAL
        elif urgency > 0.3:
            return PatientStatus.URGENT
        return PatientStatus.STABLE
    
    def _get_chief_complaint(self, urgency: float) -> str:
        if urgency > 0.7:
            return "Severe chest pain radiating to left arm"
        elif urgency > 0.3:
            return "Persistent fever and cough for 3 days"
        return "Mild headache after work"
    
    def _get_medical_history(self, urgency: float) -> List[str]:
        history = []
        if urgency > 0.5:
            history.append("Hypertension")
        if urgency > 0.6:
            history.append("Diabetes Type 2")
        if random.random() > 0.8:
            history.append("Previous hospitalization")
        return history     