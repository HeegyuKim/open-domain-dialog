from typing import Any, List, Optional
from ..base import dataset_utils
import torch


def prepare_batch(
    tokenizer: Any,
    texts: List[str],
    max_length: int,
    device: str,
    truncation_side: str = "right",
    padding_side: str = "right",
    add_bos_token: bool = False,
    add_eos_token: bool = False,
    label_for_pad_token_id: int = -100,
    return_labels: bool = True,
):
    ids = tokenizer(texts, padding=False, truncation=False, add_special_tokens=False)[
        "input_ids"
    ]
    truncate_len = min(dataset_utils.get_longest_length(ids), max_length) + 1
    pad_size = dataset_utils.pad_size_to_multiple_of(truncate_len - 1, 8) + 1

    if add_bos_token:
        truncate_len -= 1
    if add_eos_token:
        truncate_len -= 1

    new_ids = []
    masks = []
    labels = []

    for item in ids:
        # truncate 보다 긴 녀석은 eos 토큰 추가하지 않음!!
        is_truncated = len(item) > truncate_len
        real_truncate_len = (
            truncate_len + 1 if is_truncated and add_eos_token else truncate_len
        )
        item = dataset_utils.truncate(
            item,
            real_truncate_len,
            truncation_side=truncation_side,
            prefix_value=tokenizer.bos_token_id if add_bos_token else None,
            postfix_value=tokenizer.eos_token_id
            if add_eos_token and not is_truncated
            else None,
        )
        item_len = len(item)
        mask = [1] * (item_len) + [0] * (pad_size - item_len)
        label = item[1:] + [label_for_pad_token_id] * (pad_size - item_len)

        item = dataset_utils.pad(
            item,
            pad_size,
            padding_side=padding_side,
            padding_value=tokenizer.pad_token_id,
        )

        new_ids.append(item[:-1])
        labels.append(label)
        masks.append(mask[:-1])

    out = {
        "input_ids": torch.tensor(new_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
    }

    if return_labels:
        out["labels"] = torch.tensor(labels, dtype=torch.long, device=device)

    return out
