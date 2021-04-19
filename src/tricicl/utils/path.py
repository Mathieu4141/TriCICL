from pathlib import Path
from typing import Union


def safe_path(p: Union[str, Path]):
    p = Path(p)
    p.mkdir(exist_ok=True, parents=True)
    return p
