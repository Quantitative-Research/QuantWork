import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from Markowitz.dashboard_callbacks import app

if __name__ == "__main__":
    app.run(debug=True, port=8050)