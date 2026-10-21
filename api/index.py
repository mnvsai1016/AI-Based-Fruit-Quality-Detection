import os
import sys
from pathlib import Path

# Add api directory to sys.path so local imports work seamlessly in all serverless environments
API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

# Import the FastAPI application from api/main.py
from main import app

# Expose both app and handler for universal Vercel Python runtime compatibility
handler = app
