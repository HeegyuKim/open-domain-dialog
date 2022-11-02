from typing import Callable, List, Optional, Union
from ..base.task import BaseTask
from ..base.collator import ListCollator
from ..base import dataset_utils
from .. import metric, loss
from ..simctg.loss import SimCTGLoss
from . import utils
from ..simctg.loss import SimCTGLoss

from omegaconf import DictConfig
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import wandb
import torch
import torch.nn.functional as F


def get_loss_fn(config):
    loss_fn = config.model.get("loss_fn", "cross_entropy")
    loss_params = config.model.get("loss_params", {})
    if loss_fn == "focal":
        return loss.FocalLoss(**loss_params)
    elif loss_fn == "simctg":
        return SimCTGLoss(
            0.5, 
            51200,
            3
        )
    elif loss_fn == "cross_entropy":
        return loss.CrossEntropyLoss()
    elif loss_fn == "simctg":
        return loss.SimCTGLoss(
            1,
            51200,
            3
        )
    else:
        raise Exception(f"알 수 없는 loss function {loss_fn}")


class GPTTask(BaseTask):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.tokenizer = GPT2TokenizerFast.from_pretrained(config.model.plm)
        self.model = GPT2LMHeadModel.from_pretrained(config.model.plm)

        self.tokenizer.add_special_tokens(
            config.tokenizer.get("add_special_tokens", {})
        )
        self.loss_fn = get_loss_fn(config)

    def get_train_collator(self) -> Callable:
        return ListCollator()

    def get_eval_collator(self) -> Callable:
        return ListCollator()

    def step(self, texts):
        batch = utils.prepare_batch_v2(
            self.tokenizer,
            texts,
            self.config.model.max_seq_len,
            self.device
        )

        # labels = batch.pop("labels")
        out = self.model(**batch)
        return out.loss
        # logits = out.logits[..., :-1, :].contiguous()
        # labels = labels[..., 1:].contiguous()

        if loss_name == "simctg":
            loss = self.loss_fn(
                out.last_hidden_state, out.logits, batch["input_ids"], labels
            )
        else:
            loss = self.loss_fn(out.logits, labels)

        # labels = batch.pop("labels")
        # logits = out.logits[..., :-1, :].contiguous()
        # labels = labels[..., 1:].contiguous()
        
        # else:
            # loss = self.loss_fn(logits, labels)
            # loss = torch.nn.CrossEntropyLoss()(logits, labels)
            # loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss

    def generate(
        self,
        prompt: Union[str, List[str]],
        return_with_prompt: bool = False,
        min_new_tokens: int = 0,
        max_prompt_length: Optional[int] = None,
        **kwargs,
    ):
        X = self.tokenizer(
            prompt, padding=False, truncation=False, add_special_tokens=False
        )["input_ids"]
        X = [
            dataset_utils.truncate(
                x,
                max_len=self.config.model.max_seq_len
                if max_prompt_length is None
                else max_prompt_length,
                truncation_side="left",
            )
            for x in X
        ]
        prompt_lens = [len(x) for x in X]
        max_prompt_len = max(prompt_lens)
        X = [
            dataset_utils.pad(
                x, max_prompt_len, "left", padding_value=self.tokenizer.pad_token_id
            )
            for x in X
        ]
        X = torch.tensor(X, dtype=torch.long, device=self.device)

        if min_new_tokens > 0:
            kwargs["min_length"] = min_new_tokens + max_prompt_len

        generations = self.model.generate(X, **kwargs).tolist()
        if not return_with_prompt:
            generations = [g[max_prompt_len:] for p, g in zip(prompt_lens, generations)]

        generations = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
        return generations

    def _join_uttrs(self, uttrs, speaker_ids=None):
        sep = "\n"
        if speaker_ids is not None:
            uttrs = [
                "".join([str(sid) + " : " + u + sep for sid, u in zip(sids, us)])
                for sids, us in zip(speaker_ids, uttrs)
            ]
        else:
            uttrs = ["".join([u + sep for u in us]) for us in uttrs]
        return uttrs

    def training_step(self, batch, batch_idx):
        loss = self.step(batch["dialog"])
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch["context"])
        texts = self._join_uttrs(
            [
                (c + "\n" + r).split("\n")
                for c, r in zip(batch["context"], batch["response"])
            ]
        )
        loss = self.step(texts)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        if batch_idx < self.config.logger.get("val_sample_batches", 1):
            context = self._join_uttrs([c.split("\n") for c in batch["context"]])
            # response = [r + self.tokenizer.eos_token for r in batch["response"]]
            response = [r + "\n" for r in batch["response"]]
            params = self.config.logger.get("val_sample_generation_params", {})

            samples = self.generate(context, **params)

            self.log(
                "val_levenshtein_dist",
                metric.levenshtein_batch(batch["response"], samples),
                on_epoch=True,
                batch_size=batch_size,
            )
            return {
                "context": context,
                "response": response,
                "prediction": samples,
            }
        else:
            return None

    def validation_epoch_end(self, outputs) -> None:
        if wandb.run is None:
            print(outputs)
            return

        table = wandb.Table(["context", "response", "prediction"])

        for output in outputs:
            if output is None:
                continue

            for c, r, p in zip(
                output["context"], output["response"], output["prediction"]
            ):
                table.add_data(c, r, p)

        wandb.log({"val_sample": table})
