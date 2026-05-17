import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from usage.Markowitz.dashboard_callbacks import app

if __name__ == "__main__":
    app.run(debug=True, port=8050)