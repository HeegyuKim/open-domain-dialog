from typing import Any, List, Optional
from ..base import dataset_utils
import torch


def prepare_batch(
    tokenizer: Any,
    contexts: List[str],
    responses: List[str],
    encoder_max_length: int,
    decoder_max_length: int,
    device: str,
    label_for_pad_token_id: int = -100,
):

    encoder_inputs = dataset_utils.tokenize_truncate_pad(
        tokenizer,
        contexts,
        max_length=encoder_max_length,
        device=device,
        truncation_side="left",
        add_special_tokens=False,
    )
    decoder_inputs = dataset_utils.tokenize_truncate_pad(
        tokenizer,
        responses,
        decoder_max_length - 2,
        device=device,
        truncation_side="right",
        add_special_tokens=True,
        is_label=True,
    )

    inputs = encoder_inputs
    inputs["decoder_input_ids"] = decoder_inputs["input_ids"][:, :-1]
    inputs["decoder_attention_mask"] = decoder_inputs["attention_mask"][:, :-1]

    dec_labels = decoder_inputs["input_ids"][:, 1:]
    inputs["labels"] = dec_labels.masked_fill(
        dec_labels == tokenizer.pad_token_id, label_for_pad_token_id
    )

    return inputs
