import os

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import trange

from src.embeddings import convert_sequence_to_tensor
from src.transformer import BumbleBee
from src.utils import length_ok, make_std_mask

d_model = 512

# 1) Toy parallel data (you can add more examples)
# input_sentences = ["Autobots, transform and roll out"]
# target_sentences = ["<start> Autobots, transformez-vous et déployez-vous ! <eos>"]


dataset = load_dataset("bentrevett/multi30k")
MAX_SAMPLES = 1000
dataset = dataset["train"].select(range(MAX_SAMPLES))
dataset = dataset.filter(length_ok)

input_sentences = []
target_sentences = []

for ex in dataset:
    input_sentences.append(ex["de"])
    target_sentences.append(ex["en"])

input_ids, target_ids, vocab, tokenizer = convert_sequence_to_tensor(
    input_sequence=input_sentences,
    output_sequence=target_sentences,
)

# 2) Build decoder input (tgt_in) and prediction targets (tgt_out)
#    teacher forcing: predict token t+1 given tokens <= t
tgt_in = target_ids[:, :-1]  # fed to decoder
tgt_out = target_ids[:, 1:]  # ground truth


pad_id = tokenizer.pad_token_id
tgt_mask = make_std_mask(tgt_in, pad_id)

# 3) Model + loss + optimizer
model = BumbleBee(vocab_size=len(vocab), d_model=d_model, n_blocks=2, dropout=0.1)
criterion = nn.NLLLoss(ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

BATCH_SIZE = 20
EPOCHS = 10
N = input_ids.size(0)

for epoch in range(EPOCHS):
    perm = torch.randperm(N)
    input_ids = input_ids[perm]
    tgt_in = tgt_in[perm]
    tgt_out = tgt_out[perm]

    for i in trange(0, N, BATCH_SIZE, desc=f"epoch {epoch}"):
        batch_src = input_ids[i : i + BATCH_SIZE]
        batch_tgt_in = tgt_in[i : i + BATCH_SIZE]
        batch_tgt_out = tgt_out[i : i + BATCH_SIZE]

        batch_mask = make_std_mask(batch_tgt_in, pad_id)

        optimizer.zero_grad()
        log_probs = model(batch_src, batch_tgt_in, batch_mask)

        loss = criterion(
            log_probs.reshape(-1, log_probs.size(-1)),
            batch_tgt_out.reshape(-1),
        )

        loss.backward()
        optimizer.step()


os.makedirs("checkpoints", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
    },
    "checkpoints/bumblebee.pt",
)
