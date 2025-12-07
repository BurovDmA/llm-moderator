import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def download_data() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_url = (
        "hf://datasets/phucthaiv02/" "Jigsaw-Agile-Community-Rules-Classification/"
    )

    splits = {
        "train": "train.csv",
        "test": "test.csv",
    }

    for filename in splits.values():
        path = raw_dir / filename
        if path.exists():
            logger.info("%s already exists, skipping.", filename)
            continue

        logger.info("Downloading %s from HF...", filename)
        df = pd.read_csv(base_url + filename)
        df.to_csv(path, index=False)

    logger.info("Raw data downloaded to %s", raw_dir)


if __name__ == "__main__":
    download_data()
