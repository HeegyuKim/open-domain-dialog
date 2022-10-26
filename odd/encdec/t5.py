from typing import Callable, List, Union
from ..base.task import BaseTask
from ..base.collator import ListCollator
from ..base import dataset_utils
from .utils import prepare_batch
from .. import metric, loss

from omegaconf import DictConfig
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import wandb
import torch
import torch.nn.functional as F


def get_loss_fn(config):
    loss_fn = config.model.get("loss_fn", "cross_entropy")
    loss_params = config.model.get("loss_params", {})
    if loss_fn == "focal":
        return loss.FocalLoss(**loss_params)
    elif loss_fn == "cross_entropy":
        return loss.CrossEntropyLoss()
    else:
        raise Exception(f"알 수 없는 loss function {loss_fn}")



class T5Task(BaseTask):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.tokenizer = T5TokenizerFast.from_pretrained(config.model.plm)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model.plm)

        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        self.loss_fn = get_loss_fn(config)

    def get_train_collator(self) -> Callable:
        return ListCollator()

    def get_eval_collator(self) -> Callable:
        return ListCollator()

    def step(self, batch):
        context = batch["context"]
        response = batch["response"]
        
        batch = prepare_batch(
            tokenizer=self.tokenizer, 
            contexts=context,
            responses=response,
            encoder_max_length=self.config.model.encoder_max_length,
            decoder_max_length=self.config.model.decoder_max_length,
            device=self.device
            )

        labels = batch.pop("labels")
        logits = self.model(**batch).logits
        loss = self.loss_fn(logits, labels)

        # cpulabels = labels.cpu()
        # print(self.tokenizer.batch_decode(cpulabels.masked_fill(cpulabels == -100, 0)))

        return loss

    def generate(self, prompt: Union[str, List[str]], **kwargs):
        input_ids = dataset_utils.tokenize_truncate_pad(
            self.tokenizer, 
            prompt,
            max_length=self.config.model.encoder_max_length,
            truncation_side="left",
            device=self.device
            )["input_ids"]
        generations = self.model.generate(
            input_ids, 
            bos_token_id=self.tokenizer.bos_token_id,
            **kwargs
            )
        generations = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
        return generations

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch["context"])
        loss = self.step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        if batch_idx < self.config.logger.get("val_sample_batches", 1):
            params = self.config.logger.get("val_sample_generation_params", {})
            samples = self.generate(batch["context"], **params)

            self.log("val_levenshtein_dist", metric.levenshtein_batch(batch["response"], samples), on_epoch=True, batch_size=batch_size)
            return {
                "context": batch["context"],
                "response": batch["response"],
                "prediction": samples
            }
        else:
            return None

    def validation_epoch_end(self, outputs) -> None:
        if wandb.run is None:
            return

        table = wandb.Table(["context", "response", "prediction"])

        for output in outputs:
            if output is None:
                continue

            for c, r, p in zip(output["context"], output["response"], output["prediction"]):
                table.add_data(c, r, p)


        wandb.log({"val_sample": table})
