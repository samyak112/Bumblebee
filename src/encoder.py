import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork
from .norm import LayerNorm, SubLayerConnection
from .utils import create_copy


class Encoder(nn.Module):
    def __init__(self, d_model: int = 8, n_blocks: int = 6, dropout: float = 0.1) -> None:
        super().__init__()

        self.encoder_blocks = create_copy(
            EncoderLayer(d_model=d_model, n_heads=d_model // 2, dropout=dropout),
            n_blocks,
        )
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        for block in self.encoder_blocks:
            x = block.forward(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, dropout=dropout, n_heads=n_heads)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_model * 4, dropout=dropout)
        self.sublayers = create_copy(SubLayerConnection(d_model, dropout), 2)

    def forward(self, x, mask=None):
        """
        Looks trivial to pass the same parameter x three times to MHA but its necessary because
        in the decoder we will have different inputs for key and value
        """

        x = self.sublayers[0](x, lambda x: self.mha(x, x, x, mask))
        return self.sublayers[1](x, self.ffn)
