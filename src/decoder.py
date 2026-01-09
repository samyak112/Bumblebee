import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork
from .norm import LayerNorm, SubLayerConnection
from .utils import create_copy


class Decoder(nn.Module):
    def __init__(self, d_model, n_blocks, dropout: float = 0.1) -> None:
        super().__init__()
        self.decoder_blocks = create_copy(
            DecoderBlock(d_model=d_model, n_heads=d_model // 2, dropout=dropout),
            n_blocks,
        )
        self.norm = LayerNorm(features=d_model)

    def forward(self, x, y, tgt_mask=None):
        for block in self.decoder_blocks:
            x = block.forward(x, y, tgt_mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int = 8, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.sub_layers = create_copy(SubLayerConnection(size=d_model, dropout=dropout), 3)
        self.masked_mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.cross_mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=4 * d_model)

    def forward(self, encoder_embedding, x, target_mask):
        """
        this is an autoregressive step, we run MHA two times in decoder because first MHA
        gets an updated target_embedding because a new token is added into it on each iteration
        and then we create a new target_embedding, again get it contextualized and then pass it with
        the memory of encoder embedding to get the next token until the sentence is ended
        """

        x = self.sub_layers[0](x, lambda x: self.masked_mha(x, x, x, mask=target_mask))
        x = self.sub_layers[1](x, lambda x: self.cross_mha(x, encoder_embedding, encoder_embedding))
        x = self.sub_layers[2](x, self.ffn)
        return x
