import torch.nn as nn
from torch import Tensor

from .src.decoder import Decoder
from .src.embeddings import Embeddings, convert_sequence_to_tensor, get_positional_encoding
from .src.encoder import Encoder


class BumbleBee(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model: int = 8,
        # same number of encoder and decoder blocks
        n_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.encoder = Encoder(d_model=d_model, n_blocks=n_blocks, dropout=dropout)
        self.decoder = Decoder(d_model=d_model, n_blocks=n_blocks, dropout=dropout)
        position = get_positional_encoding(d_model=d_model)
        self.register_buffer("pe", position)

    def forward(self, src_ids: Tensor, tgt_ids: Tensor):
        input_embeddings = self.embedding(src_ids)
        output_embeddings = self.embedding(tgt_ids)

        # Add positional encoding

        """
            Example

            # Suppose batch size = 1, seq_len = 3, embedding dim = 4
            input_embeddings = torch.tensor([[
                [1.0, 2.0, 3.0, 4.0],  # token 0
                [5.0, 6.0, 7.0, 8.0],  # token 1
                [9.0, 10.0, 11.0, 12.0]  # token 2
            ]])  # shape: [1, 3, 4]

            # Positional encoding for 3 positions, d_model = 4
            pe = torch.tensor([[
                [0.1, 0.2, 0.3, 0.4],  # position 0
                [0.5, 0.6, 0.7, 0.8],  # position 1
                [0.9, 1.0, 1.1, 1.2]   # position 2
            ]])  # shape: [1, 3, 4]

            Input Embeddings + PE:
            tensor([[[ 1.1000,  2.2000,  3.3000,  4.4000],
                    [ 5.5000,  6.6000,  7.7000,  8.8000],
                    [ 9.9000, 11.0000, 12.1000, 13.2000]]])
        """
        input_embeddings = input_embeddings + self.pe[:, : input_embeddings.size(1), :]
        output_embeddings = output_embeddings + self.pe[:, : output_embeddings.size(1), :]

        encoder_output = self.encoder(input_embeddings)
        decoder_output = self.decoder(encoder_output, output_embeddings)


d_model = 8

input_sentence = ["Autobots, transform and roll out"]
target_sentence = ["Autobots, transformez-vous et déployez-vous !"]

input_ids_tensor, output_ids_tensor, vocab_size = convert_sequence_to_tensor(
    input_sequence=input_sentence,
    output_sequence=target_sentence,
)

transformer = BumbleBee(vocab_size=vocab_size, d_model=d_model, n_blocks=6, dropout=0.1)
transformer(input_ids_tensor, output_ids_tensor)
