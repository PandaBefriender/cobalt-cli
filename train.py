"""
train.py

TinyGPT Training Script

Responsibilities
----------------
1. Read dataset
2. Train tokenizer
3. Build training sequences
4. Train TinyGPT
5. Save checkpoint
6. Generate sample text
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    DATASET_PATH,
    CONTEXT_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    MODEL_PATH,
)

from tokenizer import Tokenizer
from model import TinyGPT


# ==========================================================
# Device
# ==========================================================

if DEVICE == "cuda" and not torch.cuda.is_available():
    print("CUDA not found. Falling back to CPU.")
    DEVICE = "cpu"

device = torch.device(DEVICE)

print(f"\nUsing device: {device}\n")


# ==========================================================
# Read Dataset
# ==========================================================

dataset_path = Path(DATASET_PATH)

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Could not find {DATASET_PATH}"
    )

text = dataset_path.read_text(
    encoding="utf-8"
)

print(f"Loaded {len(text):,} characters")


# ==========================================================
# Tokenizer
# ==========================================================

tokenizer = Tokenizer()

print("Training tokenizer...")

tokenizer.train(text)

tokenizer.save()

print(f"Vocabulary Size: {tokenizer.vocab_size:,}")

tokens = tokenizer.encode(text)

print(f"Total Tokens: {len(tokens):,}")


# ==========================================================
# Dataset
# ==========================================================

class TinyGPTDataset(Dataset):

    def __init__(self, tokens):

        self.inputs = []
        self.targets = []

        for i in range(
            len(tokens) - CONTEXT_LENGTH
        ):

            x = tokens[
                i:i + CONTEXT_LENGTH
            ]

            y = tokens[
                i + 1:i + CONTEXT_LENGTH + 1
            ]

            self.inputs.append(x)
            self.targets.append(y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        x = torch.tensor(
            self.inputs[index],
            dtype=torch.long
        )

        y = torch.tensor(
            self.targets[index],
            dtype=torch.long
        )

        return x, y


dataset = TinyGPTDataset(tokens)

print(f"Training Samples: {len(dataset):,}")

# ==========================================================
# Train / Validation Split
# ==========================================================

train_size = int(len(dataset) * 0.90)
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training Samples   : {len(train_dataset):,}")
print(f"Validation Samples : {len(val_dataset):,}")


# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)


# ==========================================================
# Build Model
# ==========================================================

model = TinyGPT(
    vocab_size=tokenizer.vocab_size
)

model = model.to(device)

print(model)

print(
    f"\nTrainable Parameters: "
    f"{model.count_parameters():,}\n"
)


# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.CrossEntropyLoss()


# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

# ==========================================================
# Resume Training
# ==========================================================

RESUME = False

start_epoch = 0


def load_checkpoint(path):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Previous loss: "
        f"{checkpoint['loss']:.4f}"
    )

    return checkpoint["epoch"]


if RESUME:

    if Path(MODEL_PATH).exists():

        start_epoch = load_checkpoint(
            MODEL_PATH
        )

    else:

        print(
            "No checkpoint found. Starting fresh."
        )


# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)


# ==========================================================
# Checkpoint Helpers
# ==========================================================

def save_checkpoint(epoch, loss):

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "vocab_size": tokenizer.vocab_size,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(f"\nModel saved -> {MODEL_PATH}\n")


def evaluate():

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1),
            )

            total_loss += loss.item()

    model.train()

    return total_loss / max(len(val_loader), 1)

# ==========================================================
# Training Loop
# ==========================================================

print("\nStarting Training...\n")

best_val_loss = float("inf")

for epoch in range(start_epoch, EPOCHS):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        leave=True,
    )

    for batch_index, (x, y) in enumerate(progress):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x)

        # Reshape for CrossEntropyLoss
        logits = logits.view(
            -1,
            tokenizer.vocab_size
        )

        targets = y.view(-1)

        loss = criterion(
            logits,
            targets
        )

        # Backpropagation
        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        average_loss = running_loss / (batch_index + 1)

        progress.set_postfix(
            loss=f"{average_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    scheduler.step()

    train_loss = running_loss / len(train_loader)

    val_loss = evaluate()

    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Training Loss : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    # Save best checkpoint
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_checkpoint(
            epoch + 1,
            val_loss
        )

# ==========================================================
# Text Generation Test
# ==========================================================

def generate_sample(prompt, max_tokens=50):

    model.eval()

    encoded = tokenizer.encode(prompt)

    if len(encoded) == 0:
        encoded = [0]

    tokens = torch.tensor(
        encoded,
        dtype=torch.long
    ).unsqueeze(0)

    tokens = tokens.to(device)

    generated = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        temperature=0.8,
    )

    generated = generated[0].tolist()

    text = tokenizer.decode(
        generated
    )

    return text


# ==========================================================
# Test Generation
# ==========================================================

print("\nTesting Generation...\n")

prompts = [
    "the",
    "once",
    "a",
]


for prompt in prompts:

    output = generate_sample(
        prompt
    )

    print(
        f"Prompt: {prompt}"
    )

    print(
        f"Output: {output}"
    )

    print("-" * 50)


# ==========================================================
# Final Save
# ==========================================================

save_checkpoint(
    EPOCHS,
    best_val_loss
)


print("\nTraining Complete!")
print(
    f"Best Validation Loss: {best_val_loss:.4f}"
)

# ==========================================================
# Reproducibility
# ==========================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)