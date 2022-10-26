from typing import List, Any, Optional
import torch


def normalize_weights(weights):
    if weights is None:
        return None

    s = sum(weights)
    return [w / s for w in weights]


def pad_size_to_multiple_of(pad_size: int, multiple_of: int):
    if pad_size % multiple_of == 0:
        return pad_size
    else:
        return multiple_of * ((pad_size // multiple_of) + 1)


def truncate(
    arr: List,
    max_len: int,
    truncation_side: str = "right",
    prefix_value: Optional[Any] = None,
    postfix_value: Optional[Any] = None,
) -> List:
    if len(arr) > max_len:
        if truncation_side == "right":
            arr = arr[:max_len]
        else:
            arr = arr[-max_len:]

    if prefix_value is not None:
        arr.insert(0, prefix_value)
    if postfix_value is not None:
        arr.append(postfix_value)

    return arr


def pad(
    arr: List, max_len: int, padding_side: str = "right", padding_value: Any = 0
) -> List:
    if len(arr) < max_len:
        p = [padding_value] * (max_len - len(arr))

        if padding_side == "right":
            arr = arr + p
        else:
            arr = p + arr

    return arr


def get_longest_length(arr):
    return max(map(len, arr))


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)


def tokenize_truncate_pad(
    tokenizer: Any,
    texts: List[str],
    max_length: int,
    device: str,
    truncation_side: str = "right",
    padding_side: str = "right",
    add_special_tokens: bool = False,
    is_label: bool = False,
):
    ids = tokenizer(texts, padding=False, truncation=False, add_special_tokens=False)[
        "input_ids"
    ]
    pad_size = min(get_longest_length(ids), max_length)
    pad_size = pad_size_to_multiple_of(pad_size, 8)

    if is_label:
        pad_size += 1

    new_ids = []
    for item in ids:
        if add_special_tokens:
            item = truncate(
                item,
                pad_size - 2,
                truncation_side=truncation_side,
                prefix_value=tokenizer.bos_token_id,
                postfix_value=tokenizer.eos_token_id,
            )
        else:
            item = truncate(item, pad_size)
        item = pad(
            item,
            pad_size,
            padding_side=padding_side,
            padding_value=tokenizer.pad_token_id,
        )
        new_ids.append(item)

    ids = torch.tensor(new_ids, dtype=torch.long, device=device)
    inputs = {"input_ids": ids, "attention_mask": (ids != tokenizer.pad_token_id).int()}

    return inputs
