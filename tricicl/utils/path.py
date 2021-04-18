from pathlib import Path


def safe_path(p: Path):
    p.mkdir(exist_ok=True, parents=True)
    return p
