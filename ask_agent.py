"""
Interactive Medical Agent - Ask about any patient
FIX: replaced non-existent UltimateDoctorAgent with RealClinicalAgent
     and replaced agent.assess_patient() with correct method signature
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Patient, VitalSigns, PatientStatus, MedicalAction

# FIX: import RealClinicalAgent + RealDataLoader instead of UltimateDoctorAgent
from inference import RealClinicalAgent, RealDataLoader


def ask_agent():
    """Interactive Q&A with the medical agent"""

    print("="*60)
    print("Loading patient data (this may take a moment)...")
    print("="*60)

    data_loader = RealDataLoader()
    agent = RealClinicalAgent(data_loader)

    print("="*60)
    print("MEDICAL AGENT - Ask me about any patient")
    print("="*60)
    print("Enter patient details (or 'quit' to exit)")
    print()

    while True:
        print("\n" + "-"*40)
        name = input("Patient name (or 'quit'): ").strip()
        if name.lower() == 'quit':
            break

        try:
            age = int(input("Age: "))
            complaint = input("Chief complaint: ")
            symptoms_input = input("Symptoms (comma separated): ")
            symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]

            hr = int(input("Heart rate (bpm): "))
            o2 = float(input("Oxygen saturation (%): "))
            bp_sys = int(input("Blood pressure systolic (mmHg): "))
            bp_dia = int(input("Blood pressure diastolic (mmHg): "))
            temp = float(input("Temperature (Celsius): "))
            rr = int(input("Respiratory rate (breaths/min) [default 16]: ") or "16")

        except (ValueError, EOFError) as e:
            print(f"Invalid input: {e}. Please enter valid numbers.")
            continue

        # Clamp vitals to valid model ranges
        hr = max(40, min(180, hr))
        o2 = max(85.0, min(100.0, o2))
        bp_sys = max(80, min(200, bp_sys))
        bp_dia = max(50, min(120, bp_dia))
        temp = max(35.0, min(40.0, temp))
        rr = max(10, min(35, rr))

        # FIX: urgency_score derived from heart_rate properly (not broken formula)
        urgency = min(1.0, max(0.0, (hr - 40) / 140.0))

        patient = Patient(
            id=name.replace(" ", "_"),
            name=name,
            age=age,
            gender="Unknown",
            symptoms=symptoms if symptoms else ["not specified"],
            vitals=VitalSigns(
                heart_rate=hr,
                blood_pressure_systolic=bp_sys,
                blood_pressure_diastolic=bp_dia,
                oxygen_saturation=o2,
                temperature=temp,
                respiratory_rate=rr
            ),
            # FIX: use PatientStatus enum correctly
            status=PatientStatus.URGENT if hr > 100 else PatientStatus.STABLE,
            urgency_score=urgency,
            time_to_deterioration=5,
            chief_complaint=complaint,
            medical_history=[]
        )

        # FIX: use assess_patient() which now exists on RealClinicalAgent
        assessment = agent.assess_patient(patient, step=0, max_steps=10)

        print("\n" + "="*50)
        print(f"AGENT ASSESSMENT FOR {name.upper()}")
        print("="*50)
        print(f"Diagnosis        : {assessment.get('diagnosis', 'Analyzing...')}")
        print(f"Risk Level       : {assessment.get('risk_level', 'Unknown')}")
        print(f"Recommended Action: {assessment['action'].upper()}")
        print(f"Confidence       : {assessment['confidence']*100:.0f}%")
        print(f"NEWS Score       : {assessment.get('news_score', 'N/A')}")
        print(f"\nReasoning: {assessment.get('reasoning', 'No reasoning available')}")
        print("="*50)

    print("\nGoodbye!")


if __name__ == "__main__":
    ask_agent()