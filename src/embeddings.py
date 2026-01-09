import math

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer


def convert_sequence_to_tensor(input_sequence: list[str], output_sequence: list[str]) -> tuple[Tensor, Tensor, int]:
    if len(input_sequence) != len(output_sequence):
        raise Exception("Input and Output Sequence length must be same")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    vocab = tokenizer.get_vocab()

    input_ids = tokenizer(
        input_sequence,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    target_ids = tokenizer(
        output_sequence,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return input_ids["input_ids"], target_ids["input_ids"], vocab, tokenizer


def get_positional_encoding(d_model, max_len=100):
    """
    Generates the positional encoding matrix.

    Args:
        max_len (int): The maximum sequence length.
        d_model (int): The dimension of the model embedding.

    Returns:
        np.ndarray: A matrix of shape (max_len, d_model) with positional encodings.
    """
    # Create a matrix to store the positional encodings
    pe = torch.zeros((max_len, d_model))

    # Create a column vector representing the positions [0, 1, ..., max_len-1]
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    # Apply sine to even indices in the array; 2i
    pe[:, 0::2] = torch.sin(position * div_term)
    # Apply cosine to odd indices in the array; 2i + 1
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # [1, max_len, d_model] batch dimension

    return pe


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
