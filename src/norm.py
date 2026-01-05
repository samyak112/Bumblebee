import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    """
        We apply residual connections separately to MHA and FFN so that the
        FFN is not forced to operate only on the fully transformed output of
        MHA. The residual path preserves the input representation, allowing
        each sublayer to learn a refinement rather than an irreversible
        transformation.
    """

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
