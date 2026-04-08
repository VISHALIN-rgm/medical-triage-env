"""
Real Medical Data Loader for Medical Triage Environment
Loads actual patient data from MIMIC-IV-ED database via Hugging Face
"""

from datasets import load_dataset
import random
from typing import Dict, List, Optional

class RealMedicalDataLoader:
    """
    Loads real anonymized patient data from actual emergency departments
    Source: MIMIC-IV-ED database via Hugging Face
    """
    
    def __init__(self):
        self.dataset = None
        self.source = "Synthetic Fallback"
        self.total_records = 0
        self.low_risk_patients = []
        self.medium_risk_patients = []
        self.high_risk_patients = []
        self._load_real_data()
        self._categorize_patients()
    
    def _fahrenheit_to_celsius(self, fahrenheit):
        if fahrenheit is None or fahrenheit != fahrenheit:
            return 37.0
        try:
            f = float(fahrenheit)
            if f > 50:
                return max(35.0, min(40.0, (f - 32) * 5 / 9))
            return max(35.0, min(40.0, f))
        except:
            return 37.0
    
    def _safe_float(self, value, default=70.0):
        if value is None or value == 'uta' or value == '' or value != value:
            return default
        try:
            return float(value)
        except:
            return default
    
    def _safe_int(self, value, default=3):
        if value is None or value == 'uta' or value == '':
            return default
        try:
            return int(float(value))
        except:
            return default
    
    def _load_real_data(self):
        """Load real patient data from Hugging Face"""
        print("\n" + "="*60)
        print("📊 LOADING REAL PATIENT DATA")
        print("="*60)
        
        try:
            print("Connecting to Hugging Face...")
            self.dataset = load_dataset("dischargesum/triage", split="train")
            self.total_records = len(self.dataset)
            self.source = "MIMIC-IV-ED (Real Emergency Department Data)"
            print(f"✅ Loaded {self.total_records:,} REAL patient records!")
            print(f"   Source: Beth Israel Deaconess Medical Center")
        except Exception as e:
            print(f"⚠️ Failed to load real data: {e}")
            self._generate_synthetic_fallback()
    
    def _generate_synthetic_fallback(self):
        """Generate synthetic data as fallback"""
        self.source = "Synthetic Data"
        self.total_records = 1000
        print(f"⚠️ Using {self.total_records} synthetic records")
        
        # Generate synthetic patients
        for i in range(300):
            self.low_risk_patients.append(self._create_synthetic_patient("low"))
        for i in range(400):
            self.medium_risk_patients.append(self._create_synthetic_patient("medium"))
        for i in range(300):
            self.high_risk_patients.append(self._create_synthetic_patient("high"))
    
    def _create_synthetic_patient(self, risk_type):
        if risk_type == "low":
            return {
                "chief_complaint": "Mild headache",
                "symptoms": ["mild pain", "fatigue"],
                "vitals": {"heart_rate": 72, "oxygen_saturation": 98.0, "temperature": 36.8,
                          "blood_pressure_systolic": 118, "blood_pressure_diastolic": 78, "respiratory_rate": 14},
                "urgency_score": 0.2, "acuity": 4, "source": "synthetic", "name": "Low Risk Patient"
            }
        elif risk_type == "medium":
            return {
                "chief_complaint": "Fever and cough",
                "symptoms": ["fever", "cough"],
                "vitals": {"heart_rate": 102, "oxygen_saturation": 94.0, "temperature": 38.5,
                          "blood_pressure_systolic": 125, "blood_pressure_diastolic": 85, "respiratory_rate": 20},
                "urgency_score": 0.6, "acuity": 3, "source": "synthetic", "name": "Medium Risk Patient"
            }
        else:
            return {
                "chief_complaint": "Chest pain",
                "symptoms": ["chest pain", "shortness of breath"],
                "vitals": {"heart_rate": 118, "oxygen_saturation": 88.0, "temperature": 37.2,
                          "blood_pressure_systolic": 85, "blood_pressure_diastolic": 55, "respiratory_rate": 28},
                "urgency_score": 0.85, "acuity": 1, "source": "synthetic", "name": "High Risk Patient"
            }
    
    def _categorize_patients(self):
        """Categorize real patients by risk level"""
        if not self.dataset:
            return
        
        print("📊 Categorizing patients by risk level...")
        
        for idx, record in enumerate(self.dataset):
            acuity = self._safe_int(record.get('acuity', 3))
            hr = self._safe_float(record.get('heartrate', 80))
            o2 = self._safe_float(record.get('o2sat', 98))
            sbp = self._safe_float(record.get('sbp', 120))
            
            is_high = (o2 < 90 or sbp < 90 or hr > 120)
            is_low = (acuity >= 4 and not is_high)
            
            temp_f = self._safe_float(record.get('temperature', 98.6))
            temp_c = self._fahrenheit_to_celsius(temp_f)
            
            patient_record = {
                "chief_complaint": record.get('chiefcomplaint', 'Unknown')[:80],
                "symptoms": self._extract_symptoms(record.get('chiefcomplaint', '')),
                "vitals": {
                    "heart_rate": int(hr),
                    "oxygen_saturation": float(o2),
                    "temperature": float(temp_c),
                    "blood_pressure_systolic": int(sbp),
                    "blood_pressure_diastolic": int(self._safe_float(record.get('dbp', 80))),
                    "respiratory_rate": int(self._safe_float(record.get('resprate', 16)))
                },
                "urgency_score": 1.0 - ((acuity - 1) / 4),
                "acuity": acuity,
                "source": "real",
                "name": f"Patient_{idx}"
            }
            
            if is_high:
                self.high_risk_patients.append(patient_record)
            elif is_low:
                self.low_risk_patients.append(patient_record)
            else:
                self.medium_risk_patients.append(patient_record)
        
        print(f"   LOW risk: {len(self.low_risk_patients)} patients")
        print(f"   MEDIUM risk: {len(self.medium_risk_patients)} patients")
        print(f"   HIGH risk: {len(self.high_risk_patients)} patients")
    
    def _extract_symptoms(self, chief_complaint):
        symptoms = []
        chief_lower = chief_complaint.lower() if chief_complaint else ""
        symptom_map = {
            "chest pain": ["chest", "cardiac"],
            "shortness of breath": ["breath", "sob"],
            "fever": ["fever"],
            "headache": ["headache"],
            "nausea": ["nausea", "vomiting"],
            "cough": ["cough"],
            "abdominal pain": ["abdominal", "stomach"]
        }
        for symptom, keywords in symptom_map.items():
            if any(kw in chief_lower for kw in keywords):
                symptoms.append(symptom)
        if not symptoms and chief_complaint:
            symptoms.append(chief_complaint[:30].lower())
        return symptoms[:3]
    
    def get_balanced_patients(self, num_patients: int) -> List[Dict]:
        """Get patients with balanced risk distribution"""
        num_low = int(num_patients * 0.30)
        num_medium = int(num_patients * 0.40)
        num_high = num_patients - num_low - num_medium
        
        patients = []
        
        if self.low_risk_patients:
            patients.extend(random.sample(self.low_risk_patients, min(num_low, len(self.low_risk_patients))))
        if self.medium_risk_patients:
            patients.extend(random.sample(self.medium_risk_patients, min(num_medium, len(self.medium_risk_patients))))
        if self.high_risk_patients:
            patients.extend(random.sample(self.high_risk_patients, min(num_high, len(self.high_risk_patients))))
        
        random.shuffle(patients)
        return patients
    
    def get_random_patient(self) -> Dict:
        """Get a random patient record"""
        if self.high_risk_patients:
            return random.choice(self.high_risk_patients)
        elif self.medium_risk_patients:
            return random.choice(self.medium_risk_patients)
        else:
            return random.choice(self.low_risk_patients)
    
    def get_statistics(self):
        return {
            "total_records": self.total_records,
            "data_source": self.source,
            "low_count": len(self.low_risk_patients),
            "medium_count": len(self.medium_risk_patients),
            "high_count": len(self.high_risk_patients)
        }


# For backward compatibility
RealDataLoader = RealMedicalDataLoader