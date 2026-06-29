import sys
from pathlib import Path

# Add the project root directory to sys.path so relative imports work correctly on Vercel
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.football_ai.api import app
