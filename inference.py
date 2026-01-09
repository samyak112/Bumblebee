import torch

from src.embeddings import convert_sequence_to_tensor
from src.transformer import BumbleBee

d_model = 512

input_sentence = ["Wie geht es dir"]
target_sentence = ["<start>"]

input_ids_tensor, output_ids_tensor, vocab, tokenizer = convert_sequence_to_tensor(
    input_sequence=input_sentence,
    output_sequence=target_sentence,
)

checkpoint = torch.load("checkpoints/bumblebee.pt", map_location="cpu")

transformer = BumbleBee(
    vocab_size=len(checkpoint["vocab"]),
    d_model=512,
    n_blocks=6,
    dropout=0.1,
)

transformer.load_state_dict(checkpoint["model_state_dict"])
transformer.eval()

# Autoregressive generation
max_length = 50  # Maximum sequence length to generate
eos_token_id = tokenizer.convert_tokens_to_ids("<end>")  # Adjust based on your tokenizer

with torch.no_grad():
    generated_ids = output_ids_tensor.clone()  # Start with "<start>"

    for _ in range(max_length):
        # Forward pass with current sequence
        output = transformer(input_ids_tensor, generated_ids)

        # Get prediction for the next token (last position)
        next_token_logits = output[:, -1, :]  # Shape: (batch_size, vocab_size)
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)  # Shape: (batch_size, 1)

        # Append predicted token to the sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # Stop if EOS token is generated
        if next_token_id.item() == eos_token_id:
            break

    # Decode the generated sequence
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(decoded)
