from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from llm_moderator.data.download import download_data
from llm_moderator.data.preprocess import preprocess

logger = logging.getLogger(__name__)


def _try_dvc_pull(target_path: Path) -> bool:
    dvc_file = Path(str(target_path) + ".dvc")
    cmd: list[str]
    if dvc_file.exists():
        cmd = ["dvc", "pull", str(dvc_file)]
    else:
        cmd = ["dvc", "pull"]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("DVC pull failed (%s). Will try to build data locally.", exc)
        return False

    return target_path.exists()


def ensure_data(dvc_target: str = "data/processed") -> None:
    target_path = Path(dvc_target)

    if target_path.exists():
        return

    if _try_dvc_pull(target_path):
        return
    logger.info("Building data locally via download + preprocess")
    download_data()
    preprocess()

    if not target_path.exists():
        raise RuntimeError(
            f"Failed to prepare data at {target_path}. "
            "Tried DVC pull and local download+preprocess."
        )
