import copy

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


def create_copy(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size, device):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape, device=device),
        diagonal=1,
    ).bool()
    return subsequent_mask


"""
    std_mask (Standard Mask):
    Combines two masking operations for the Decoder's self-attention layer and is responsible for two ops:

    1. Padding Mask: Masks columns where the token is a <pad>.
    2. Subsequent Mask: Masks future positions (Upper Triangular Matrix).

    A position is only visible (1) if it is NOT padded AND is at or before the current position.

    Example: Batch with sentence "Hi" (len 2) padded to 4.
    Input: ["Hi", "End", <Pad>, <Pad>]
    Resulting Mask:
    [
    [1, 0, 0, 0],  # "Hi"  : Sees self. Future blocked.
    [1, 1, 0, 0],  # "End" : Sees "Hi", "End". Future blocked.
    [1, 1, 0, 0],  # <Pad> : Sees "Hi", "End". Can't see self (blocked by Pad Mask).
    [1, 1, 0, 0]   # <Pad> : Sees "Hi", "End". Can't see self (blocked by Pad Mask).
    ]

    Example: Normal Sentence
    Input: ["Hi", "how", "are", "you"]
    Resulting Mask:
    [
    [1, 0, 0, 0],  # "Hi"  : Sees self. Future blocked.
    [1, 1, 0, 0],  # "how" : Sees "Hi", "how". Future blocked.
    [1, 1, 1, 0],  #  are : 
    [1, 1, 1, 1]   #  you : 
    ]
"""


def make_std_mask(tgt, pad, device="cpu"):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1), device).type_as(
        tgt_mask.data,
    )
    return tgt_mask


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def length_ok(ex):
    return len(ex["de"].split()) <= 30 and len(ex["en"].split()) <= 30
