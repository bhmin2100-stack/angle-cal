from pathlib import Path
import sys

# Always run this checkout, even if .venv has an editable install from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from angle_cal.app import main


if __name__ == "__main__":
    main()
