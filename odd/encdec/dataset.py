from dataclasses import dataclass
from random import randrange
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union, List
from datasets import load_dataset, interleave_datasets
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from tokenizers import Tokenizer
import re


@dataclass
class T5LMAdaptedIterableDataset(IterableDataset):
    dataset: Dataset
    text_column: str = "text"

    min_text_length: int = 256
    split_start: float = 0.25
    split_end: float = 0.75

    def filter(self, text):
        return len(text) >= self.min_text_length

    def split_context_response(self, text: str) -> Tuple[str, str]:
        text_size = len(text)
        split_pos = randrange(
            int(text_size * self.split_start),
            int(text_size * self.split_end),
        )
        context = text[:split_pos]
        response = text[split_pos:]
        return context, response

    def __iter__(self) -> Iterator[Dict]:
        for item in self.dataset:
            text = item[self.text_column]
            if self.filter(text):
                ctx, res = self.split_context_response(text)
                yield {
                    **item,
                    "context": ctx,
                    "response": res,
                }
