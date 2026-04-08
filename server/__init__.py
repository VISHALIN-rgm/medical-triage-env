# server/__init__.py
# This file makes the server directory a Python package

from .medical_triage_env_environment import MedicalTriageEnvironment

__all__ = ['MedicalTriageEnvironment']