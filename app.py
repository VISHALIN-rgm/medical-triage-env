"""
Medical Triage Environment - Hugging Face Space Entry Point
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app
from server.app import app

# This is the app that Hugging Face Spaces will run