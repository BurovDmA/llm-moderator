from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_data(dvc_target: str = "data/processed") -> None:
    target_path = Path(dvc_target)

    if target_path.exists():
        return

    dvc_file = Path(str(target_path) + ".dvc")
    if dvc_file.exists():
        subprocess.run(["dvc", "pull", str(dvc_file)], check=True)
        return

    subprocess.run(["dvc", "pull"], check=True)
