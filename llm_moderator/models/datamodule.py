from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer_name: str,
        max_seq_length: int,
        truncation: bool,
        padding: str,
    ) -> None:
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding

        df = pd.read_csv(path)
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class LLMModeratorDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        pretrained_name: str,
        max_seq_length: int,
        batch_size: int,
        truncation: bool,
        padding: str,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.pretrained_name = pretrained_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.truncation = truncation
        self.padding = padding
        self.num_workers = num_workers

        self._train_dataset: TextDataset | None = None
        self._val_dataset: TextDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = TextDataset(
                path=self.train_path,
                tokenizer_name=self.pretrained_name,
                max_seq_length=self.max_seq_length,
                truncation=self.truncation,
                padding=self.padding,
            )
            self._val_dataset = TextDataset(
                path=self.val_path,
                tokenizer_name=self.pretrained_name,
                max_seq_length=self.max_seq_length,
                truncation=self.truncation,
                padding=self.padding,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
