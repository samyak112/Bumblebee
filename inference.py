import torch

from src.embeddings import convert_sequence_to_tensor
from src.transformer import BumbleBee

d_model = 128  # smaller model works better for 1k samples

# Example input sentence
input_sentence = ["Wie geht es dir"]
target_sentence = ["<start>"]

# Convert sequences to tensors
input_ids_tensor, output_ids_tensor, vocab, tokenizer = convert_sequence_to_tensor(
    input_sequence=input_sentence,
    output_sequence=target_sentence,
)

# Load trained model checkpoint
checkpoint = torch.load("checkpoints/bumblebee.pt", map_location="cpu")

transformer = BumbleBee(
    vocab_size=len(checkpoint["vocab"]),
    d_model=d_model,
    n_blocks=2,  # small model
    dropout=0.1,
)
transformer.load_state_dict(checkpoint["model_state_dict"])
transformer.eval()

# Autoregressive generation
max_length = 50
eos_token_id = tokenizer.convert_tokens_to_ids("<end>")
start_token_id = tokenizer.convert_tokens_to_ids("<start>")

with torch.no_grad():
    # Start generation with <start> token
    generated_ids = torch.tensor([[start_token_id]], dtype=torch.long)

    for _ in range(max_length):
        # Forward pass with current generated sequence
        output = transformer(input_ids_tensor, generated_ids)

        # Prediction for the next token (last position)
        next_token_logits = output[:, -1, :]  # (batch_size, vocab_size)

        # Temperature sampling to avoid repeated tokens
        temperature = 1.0  # increase to make output more random
        probs = torch.exp(next_token_logits / temperature)
        next_token_id = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Append predicted token
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # Stop if EOS token generated
        if next_token_id.item() == eos_token_id:
            break

    # Decode generated tokens
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(decoded)
