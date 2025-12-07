from __future__ import annotations

import subprocess

from hydra import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from llm_moderator.data.dvc_utils import ensure_data
from llm_moderator.models.datamodule import LLMModeratorDataModule
from llm_moderator.models.lightning_module import LLMModeratorModule


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def train(cfg: DictConfig) -> None:
    ensure_data(cfg.data.dvc_target)

    datamodule = LLMModeratorDataModule(
        train_path=cfg.data.train_path,
        val_path=cfg.data.val_path,
        pretrained_name=cfg.model.pretrained_name,
        max_seq_length=cfg.data.max_seq_length,
        truncation=cfg.data.truncation,
        padding=cfg.data.padding,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = LLMModeratorModule(
        pretrained_name=cfg.model.pretrained_name,
        num_labels=cfg.model.num_labels,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.betas),
        eps=cfg.train.eps,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    mlflow_logger.log_hyperparams(
        {
            "model.pretrained_name": cfg.model.pretrained_name,
            "model.num_labels": cfg.model.num_labels,
            "train.max_epochs": cfg.train.max_epochs,
            "train.learning_rate": cfg.train.learning_rate,
            "train.weight_decay": cfg.train.weight_decay,
            "train.betas": cfg.train.betas,
            "train.eps": cfg.train.eps,
            "data.batch_size": cfg.data.batch_size,
            "data.effective_batch_size": cfg.data.effective_batch_size,
            "data.max_seq_length": cfg.data.max_seq_length,
            "data.truncation": cfg.data.truncation,
            "data.padding": cfg.data.padding,
        }
    )

    mlflow_logger.experiment.set_tag(
        mlflow_logger.run_id, "git_commit", _get_git_commit()
    )

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=mlflow_logger,
    )

    trainer.fit(model=model, datamodule=datamodule)


def main() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config")
        train(cfg)


if __name__ == "__main__":
    main()
