"""
Interactive Medical Agent - Ask about any patient
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Patient, VitalSigns, PatientStatus, MedicalAction
from inference import UltimateDoctorAgent

def ask_agent():
    """Interactive Q&A with the medical agent"""
    
    agent = UltimateDoctorAgent(use_llm=False)
    
    print("="*60)
    print("🏥 MEDICAL AGENT - Ask me about any patient")
    print("="*60)
    print("Enter patient details (or 'quit' to exit)")
    print()
    
    while True:
        print("\n" + "-"*40)
        name = input("Patient name: ").strip()
        if name.lower() == 'quit':
            break
        
        try:
            age = int(input("Age: "))
            complaint = input("Chief complaint: ")
            symptoms = input("Symptoms (comma separated): ").split(',')
            hr = int(input("Heart rate: "))
            o2 = int(input("Oxygen saturation (%): "))
            bp_sys = int(input("Blood pressure (systolic): "))
            bp_dia = int(input("Blood pressure (diastolic): "))
            temp = float(input("Temperature (°C): "))
            
            # Create patient
            patient = Patient(
                id=name,
                name=name,
                age=age,
                gender="Unknown",
                symptoms=[s.strip() for s in symptoms],
                vitals=VitalSigns(
                    heart_rate=hr,
                    blood_pressure_systolic=bp_sys,
                    blood_pressure_diastolic=bp_dia,
                    oxygen_saturation=o2,
                    temperature=temp,
                    respiratory_rate=16
                ),
                status=PatientStatus.URGENT if hr > 100 else PatientStatus.STABLE,
                urgency_score=1.0 - ((max(0, min(100, (hr-60)/100)))),
                time_to_deterioration=5,
                chief_complaint=complaint
            )
            
            # Get assessment
            assessment = agent.assess_patient(patient, step=0, max_steps=10)
            
            print("\n" + "="*50)
            print(f"🤖 AGENT'S ASSESSMENT FOR {name.upper()}")
            print("="*50)
            print(f"Diagnosis: {assessment.get('diagnosis', 'Analyzing...')}")
            print(f"Risk Level: {assessment.get('risk_level', 'Unknown')}")
            print(f"Recommended Action: {assessment['action'].upper()}")
            print(f"Confidence: {assessment['confidence']*100:.0f}%")
            print(f"\nReasoning: {assessment.get('reasoning', 'No reasoning')}")
            print("="*50)
            
        except Exception as e:
            print(f"Error: {e}. Please enter valid numbers.")

if __name__ == "__main__":
    ask_agent()