import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preprocess() -> None:
    raw_path = Path("data/raw/train.csv")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    def prompt_func(x):
        return f"""
        Estimate the fact of rule violation of the following text.
        Text:
        {x['body']}
        Rule:
        {x['rule']}
        As examples:
        Positive:
        {x['positive_example_1']}
        Negative:
        {x['negative_example_1']}
        """

    df["text"] = df.apply(prompt_func, axis=1)
    df = df.rename(columns={"rule_violation": "label"})

    train_df, val_df = train_test_split(
        df[["text", "label"]], test_size=0.1, random_state=42
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    logger.info("Processed data saved to %s", out_dir)


if __name__ == "__main__":
    preprocess()
