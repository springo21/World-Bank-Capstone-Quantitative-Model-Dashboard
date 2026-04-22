import subprocess
import sys
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py"
])