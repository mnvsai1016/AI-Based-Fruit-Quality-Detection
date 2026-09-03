import os
import sys
from pathlib import Path

# Add app directory to sys.path with multiple resolution strategies
ROOT_DIR = Path(__file__).resolve().parent.parent
app_dir = ROOT_DIR / "Desing_end" / "app"
cwd_app_dir = Path.cwd() / "Desing_end" / "app"

for p in [str(app_dir), str(cwd_app_dir), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from main import app
except ImportError:
    from Desing_end.app.main import app

