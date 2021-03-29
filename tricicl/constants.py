from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
LOGS_DIR = PROJECT_DIR / "logs"
TB_DIR = LOGS_DIR / "tb"

TB_DIR.mkdir(exist_ok=True, parents=True)
