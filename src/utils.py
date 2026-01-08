import copy

import torch
import torch.nn as nn


def create_copy(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8,
    )
    return subsequent_mask == 0


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


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
        tgt_mask.data,
    )
    return tgt_mask
