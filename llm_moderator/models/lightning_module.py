from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification


class LLMModeratorModule(LightningModule):
    def __init__(
        self,
        pretrained_name: str,
        num_labels: int,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_name,
            config=config,
        )

    def forward(self, **batch: dict[str, torch.Tensor]) -> Any:
        return self.model(**batch)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            prob_pos = probs[:, 1].detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            try:
                roc_auc = roc_auc_score(y_true, prob_pos)
            except ValueError:
                roc_auc = float("nan")

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            f"{stage}_roc_auc",
            roc_auc,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        self.log(
            "debug/train_mode",
            float(self.training),
            prog_bar=True,
            on_step=True,
            logger=False,
        )

        self.model.train()
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self.model.eval()
        self._step(batch, "val")

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )
