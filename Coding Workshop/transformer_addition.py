import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Transformer for 3-Digit Addition
=================================
Trains a small attention-based transformer to add two 3-digit numbers.
Example: "123+456" -> "579"

Architecture:
  - Character-level tokenization
  - Embedding + positional encoding
  - Multi-head self-attention (causal decoder-style)
  - Feed-forward layers
  - Cross-entropy loss on output digits

Usage:
  pip install torch
  python transformer_addition.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
VOCAB       = list("0123456789+= ")   # tokens
PAD_CHAR    = " "
SEQ_LEN     = 12   # e.g. "123+456=    " (input + output padded to 12)
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 3
D_FF        = 96
DROPOUT     = 0.1
BATCH_SIZE  = 256
EPOCHS      = 50
LR          = 3e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}

def encode(s: str) -> list[int]:
    return [char2idx[c] for c in s]

def decode(ids: list[int]) -> str:
    return "".join(idx2char[i] for i in ids)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def make_example(a: int = None, b: int = None):
    """
    Returns (src_ids, tgt_ids) for a + b = c.
    Src: "AAA+BBB="  (8 chars)
    Tgt: " CCC"      (4 chars, space-padded on the left so total = 12)
    The full sequence is src+tgt, length SEQ_LEN=12.
    """
    if a is None:
        a = random.randint(0, 999)
    if b is None:
        b = random.randint(0, 999)
    c = a + b

    src = f"{a:03d}+{b:03d}="       # e.g. "123+456="
    tgt = f"{c:04d}"                 # e.g. "0579"

    full = src + tgt                 # length 12
    assert len(full) == SEQ_LEN, f"Expected {SEQ_LEN}, got {len(full)}: '{full}'"
    return encode(full)


def make_dataset(n: int):
    data = [make_example() for _ in range(n)]
    return torch.tensor(data, dtype=torch.long)   # (n, SEQ_LEN)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, seq_len):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        x = self.transformer(x, mask=mask)
        return self.head(x)   # (B, T, vocab_size)


def causal_mask(seq_len, device):
    """Upper-triangular mask so position i can only attend to positions ≤ i."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")
    print("Building dataset...")
    train_data = make_dataset(80_000).to(DEVICE)
    val_data   = make_dataset(2_000).to(DEVICE)

    model = AdditionTransformer(
        vocab_size=len(VOCAB),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, seq_len=SEQ_LEN
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    mask = causal_mask(SEQ_LEN, DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(train_data))
        total_loss = 0
        steps = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[idx[i:i+BATCH_SIZE]]   # (B, 12)
            src = batch[:, :-1]   # (B, 11) — input
            tgt = batch[:, 1:]    # (B, 11) — next-token targets

            logits = model(src, mask=mask[:-1, :-1])  # (B, 11, V)
            loss = criterion(logits.reshape(-1, len(VOCAB)), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / steps

        # Validation accuracy (on the 4 output digits only)
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, val_data, mask)
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Val digit-acc: {acc:.1%}")

    print("\nTraining complete. Running inference examples:\n")
    demo(model, mask)
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, data, mask):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            src = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(src, mask=mask[:-1, :-1])
            preds = logits.argmax(-1)
            # Only evaluate on positions 8-11 (the 4 output digits)
            correct += (preds[:, 7:] == tgt[:, 7:]).sum().item()
            total   += tgt[:, 7:].numel()
    return correct / total


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(model, a: int, b: int, mask) -> str:
    model.eval()
    # Build src: "AAA+BBB=" then autoregressively generate 4 digits
    src_str = f"{a:03d}+{b:03d}="
    ids = encode(src_str)   # length 8

    with torch.no_grad():
        for _ in range(4):
            x = torch.tensor([ids], device=DEVICE)           # (1, len)
            m = causal_mask(len(ids), DEVICE)
            logits = model(x, mask=m)                        # (1, len, V)
            next_id = logits[0, -1].argmax().item()
            ids.append(next_id)

    generated = decode(ids[8:])  # the 4 predicted digits
    return generated.strip()


def demo(model, mask):
    test_cases = [
        (123, 456),
        (999, 1),
        (500, 500),
        (0,   0),
        (371, 629),
        (100, 200),
    ]
    for a, b in test_cases:
        pred = predict(model, a, b, mask)
        actual = a + b
        status = "✓" if int(pred) == actual else "✗"
        print(f"  {status}  {a:3d} + {b:3d} = {actual:4d}  |  predicted: {pred}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = train()

    print("\n\nInteractive mode — type two numbers (e.g. '123 456') or 'quit':")
    mask = causal_mask(SEQ_LEN, DEVICE)
    while True:
        line = input("> ").strip()
        if line.lower() in ("quit", "exit", "q"):
            break
        try:
            parts = line.split()
            a, b = int(parts[0]), int(parts[1])
            if not (0 <= a <= 999 and 0 <= b <= 999):
                print("Please enter numbers between 0 and 999.")
                continue
            pred = predict(model, a, b, mask)
            print(f"  {a} + {b} = {pred}  (actual: {a+b})")
        except Exception:
            print("  Enter two integers, e.g. '123 456'")



