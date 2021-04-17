from pathlib import Path

from tricicl.utils.path import safe_path

PROJECT_DIR = Path(__file__).parent.parent
LOGS_DIR = PROJECT_DIR / "logs"
TB_DIR = safe_path(LOGS_DIR / "tb")
CSV_DIR = safe_path(LOGS_DIR / "csv")


SEEDS = [42, 51, 441390, 6584, 601065, 845929, 960932, 478844, 565362, 388222]
