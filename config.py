"""
config.py

Central configuration for TinyGPT.
All model and training hyperparameters live here.
"""

# ===========================
# Dataset
# ===========================

DATASET_PATH = "dataset.txt"

# ===========================
# Tokenizer
# ===========================

MIN_FREQUENCY = 1

# ===========================
# Model
# ===========================

CONTEXT_LENGTH = 128

EMBED_DIM = 256

NUM_HEADS = 8

NUM_LAYERS = 8

FFN_DIM = 1024

DROPOUT = 0.1

# ===========================
# Training
# ===========================

BATCH_SIZE = 32

EPOCHS = 3

LEARNING_RATE = 3e-4

DEVICE = "cuda"

# ===========================
# Saving
# ===========================

MODEL_PATH = "tinygpt.pt"