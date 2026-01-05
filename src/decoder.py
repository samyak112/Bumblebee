import torch.nn as nn

from .utils import create_copy
from .norm import SubLayerConnection,LayerNorm


class Decoder(nn.Module):
    def __init__(self, d_model, n_blocks) -> None:
        super().__init__()
        self.decoder_blocks = create_copy(DecoderLayer(d_model=d_model, n_heads=d_model // 2), n_blocks)
        self.norm = LayerNorm(features=d_model)

    def forward(self,x):
        for block in self.decoder_blocks:
            x = block.forward(x)
        
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 8, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.sub_layers = create_copy(SubLayerConnection(size=d_model,dropout=dropout),3)

    def forward(self,x):
        for layer in 

    
