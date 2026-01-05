from .embeddings import Vocabulary
from .encoder import Encoder


class Transformers:
    def __init__(
        self,
        vocab: Vocabulary,
        d_model: int = 8,
        # same number of encoder and decoder blocks
        n_blocks: int = 6,
        dropout: float = 0.1,
    ):
        self.vocab = vocab
        self.encoder = Encoder(d_model=d_model, n_blocks=n_blocks, dropout=dropout)
