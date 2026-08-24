"""
generate.py

TinyGPT Text Generation

Loads a trained checkpoint
and generates text.
"""

import torch

from config import (
    DEVICE,
    MODEL_PATH,
)

from tokenizer import Tokenizer
from model import TinyGPT


# ==========================================================
# Device
# ==========================================================

if DEVICE == "cuda" and not torch.cuda.is_available():

    print(
        "CUDA unavailable. Using CPU."
    )

    DEVICE = "cpu"


device = torch.device(DEVICE)


# ==========================================================
# Load Tokenizer
# ==========================================================

tokenizer = Tokenizer()

tokenizer.load(
    "vocab.json"
)


print(
    f"Vocabulary size: {tokenizer.vocab_size}"
)


# ==========================================================
# Load Model
# ==========================================================

model = TinyGPT(
    vocab_size=tokenizer.vocab_size
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model"]
)


model.to(device)

model.eval()


print(
    "\nTinyGPT loaded successfully!"
)

print(
    f"Parameters: {model.count_parameters():,}"
)

# ==========================================================
# Generation Function
# ==========================================================

def generate_text(
    prompt,
    max_tokens=100,
    temperature=0.8,
):

    encoded = tokenizer.encode(prompt)

    if len(encoded) == 0:

        encoded = [0]


    tokens = torch.tensor(
        encoded,
        dtype=torch.long
    )

    tokens = tokens.unsqueeze(0)

    tokens = tokens.to(device)


    output = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )


    output = output[0].tolist()


    return tokenizer.decode(output)


# ==========================================================
# Interactive Loop
# ==========================================================

print("\nType 'exit' to quit.\n")


while True:

    prompt = input(
        "You: "
    )


    if prompt.lower() == "exit":

        print(
            "Goodbye!"
        )

        break


    result = generate_text(
        prompt
    )


    print(
        "\nTinyGPT:"
    )

    print(
        result
    )

    print(
        "\n" + "-" * 60 + "\n"
    )
