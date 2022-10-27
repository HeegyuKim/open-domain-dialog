from typing import Any, Callable, Mapping, Optional, Union
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from hydra import initialize, compose

from .dataset import HuggingfaceDataset


def get_logger(config):
    name = config.get("logger", {}).get("name")

    if name == "wandb":
        return WandbLogger(name=config.run_name, project=config.project,)
    elif name == None:
        return None
    else:
        raise Exception(f"{name} is invalid logger")


class BaseTask(pl.LightningModule):
    config: DictConfig

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

    def get_optimizer(self):
        oc = self.config.optimizer

        if oc.cls == "adam":
            return optim.Adam(self.parameters(), lr=oc.learning_rate)
        if oc.cls == "adamw":
            return optim.AdamW(self.parameters(), lr=oc.learning_rate)
        else:
            raise Exception(f"{oc.cls} is unsupported optimizer type.")

    def get_scheduler(self):
        return None

    def configure_optimizers(self):
        opt = self.get_optimizer()
        sched = self.get_scheduler()
        if sched is not None:
            return [opt], [sched]
        else:
            return opt

    def get_train_dataset(self) -> Dataset:
        return HuggingfaceDataset(
            self.config.dataset.train.paths,
            self.config.dataset.train.get("weights"),
            self.config.dataset.train.get("split", "train"),
            use_auth_token=self.config.dataset.train.get("use_auth_token", False),
        )

    def get_train_collator(self) -> Callable:
        return None

    def get_eval_dataset(self) -> Dataset:
        if "validation" in self.config.dataset:
            return HuggingfaceDataset(
                self.config.dataset.validation.paths,
                weights=self.config.dataset.validation.get("weights"),
                split=self.config.dataset.validation.get("split", "test"),
                use_auth_token=self.config.dataset.train.get("use_auth_token", False),
            )
        else:
            return None

    def get_eval_collator(self) -> Callable:
        return None

    def log_dict(
        self,
        dictionary: Mapping,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        rank_zero_only: bool = False,
        prefix: Optional[str] = None,
    ) -> None:
        if prefix is not None:
            dictionary = {prefix + k: v for k, v in dictionary.items()}
        super().log_dict(
            dictionary,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            rank_zero_only,
        )

    def train_dataloader(self):
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.config.trainer.train_batch_size,
            shuffle=self.config.dataset.train.get("shuffle", True),
            num_workers=self.config.trainer.get("num_workers", 1),
            collate_fn=self.get_train_collator(),
        )

    def val_dataloader(self):
        dataset = self.get_eval_dataset()
        if dataset is not None:
            return DataLoader(
                dataset,
                batch_size=self.config.trainer.eval_batch_size,
                num_workers=self.config.trainer.get("num_workers", 1),
                collate_fn=self.get_eval_collator(),
            )
        else:
            return None

    @classmethod
    def main(cls, config_name: str):
        initialize("../../config/")
        config = compose(config_name + ".yaml")
        ckpt = config.get("checkpoint", None)

        if ckpt is None:
            task = cls(config=config)
        else:
            task = cls.load_from_checkpoint(ckpt)

        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath="./checkpoint/", filename=f"{config.project}-{config.run_name}",
        )
        trainer = pl.Trainer(
            logger=get_logger(config),
            accelerator=config.trainer.get(
                "accelerator", "gpu" if torch.cuda.is_available() else "cpu"
            ),
            devices=config.trainer.get("devices", 1),
            max_epochs=config.trainer.get("train_epochs", None),
            max_steps=config.trainer.get("train_steps", -1),
            accumulate_grad_batches=config.trainer.get("accumulate_grad_batches"),
            limit_train_batches=config.trainer.get("limit_train_batches"),
            limit_val_batches=config.trainer.get("limit_val_batches"),
            log_every_n_steps=config.trainer.get("log_every_n_steps", 1),
            val_check_interval=config.trainer.get("val_check_interval", None),
            check_val_every_n_epoch=config.trainer.get("check_val_every_n_epoch"),
            num_sanity_val_steps=config.trainer.get("num_sanity_val_steps", 0),
            strategy=config.trainer.get("strategy", None),
            gradient_clip_val=config.trainer.get("gradient_clip_val", 0),
            callbacks=[checkpoint],
        )
        trainer.fit(task, ckpt_path=config.trainer.get("resume_from_checkpoint"))
