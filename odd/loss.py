"""
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Optional


class FocalLoss:
    def __init__(
        self, gamma=0, alpha=None, size_average=True, ignore_index: Optional[int] = -100
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        input = F.log_softmax(input, dim=-1)
        if input.dim() > 2:
            input = input.view(-1, input.shape[-1])
            target = target.view(-1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target.masked_fill(mask == 0, 0)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.unsqueeze(1))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.ignore_index is not None:
            return (loss * mask.float()).sum() / mask.sum()
        else:
            return loss.mean()


class CrossEntropyLoss:
    def __init__(self, ignore_index: Optional[int] = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def __call__(self, logits, targets):
        """
        logits: [bs, seq, vocab]
        targets: [bs, seq]
        """
        # Flatten the tokens
        print("logits")
        logits = logits.view(-1, logits.size(-1))
        print("targets")
        targets = targets.view(-1)
        print("loss")
        loss = self.loss_fct(logits, targets)
        print("loss after")
        return loss
