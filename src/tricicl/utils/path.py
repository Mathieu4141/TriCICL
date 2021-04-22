from pathlib import Path
from shutil import rmtree
from typing import Union


def safe_path(p: Union[str, Path]):
    p = Path(p)
    p.mkdir(exist_ok=True, parents=True)
    return p


def empty_dir(p: Path) -> Path:
    rmtree(p, ignore_errors=True)
    return safe_path(p)
