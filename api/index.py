import sys
from pathlib import Path

# Add app directory to sys.path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "Desing_end" / "app"))

from main import app
