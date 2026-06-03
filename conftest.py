# conftest.py
import sys
from pathlib import Path

# Add project root to sys.path so `from src.eval.X import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent))
