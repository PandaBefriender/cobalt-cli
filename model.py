"""
model.py

TinyGPT
A decoder-only Transformer written completely from scratch.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CONTEXT_LENGTH,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FFN_DIM,
    DROPOUT,
)


# ============================================================
# Multi-Head Self Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Self Attention
    """

    def __init__(self):
        super().__init__()

        assert EMBED_DIM % NUM_HEADS == 0

        self.num_heads = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS

        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1))
        scores /= math.sqrt(self.head_dim)

        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = attention @ v

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, T, C)

        return self.proj(out)


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    One complete Transformer block.
    """

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attention = MultiHeadAttention()

        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ffn = FeedForward()

    def forward(self, x):

        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


# ============================================================
# TinyGPT
# ============================================================

class TinyGPT(nn.Module):
    """
    TinyGPT Decoder-Only Language Model
    """

    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        # Token Embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM
        )

        # Learned Position Embeddings
        self.position_embedding = nn.Embedding(
            CONTEXT_LENGTH,
            EMBED_DIM
        )

        self.dropout = nn.Dropout(DROPOUT)

        # Transformer Stack
        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.ln_final = nn.LayerNorm(EMBED_DIM)

        # Language Modeling Head
        self.head = nn.Linear(
            EMBED_DIM,
            vocab_size,
            bias=False
        )

        # GPT-2 Weight Tying
        self.head.weight = self.token_embedding.weight

    def forward(self, tokens):

        B, T = tokens.shape

        if T > CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence length ({T}) exceeds CONTEXT_LENGTH ({CONTEXT_LENGTH})"
            )

        positions = torch.arange(
            T,
            device=tokens.device
        )

        x = self.token_embedding(tokens)
        x = x + self.position_embedding(positions)

        x = self.dropout(x)

        x = self.blocks(x)

        x = self.ln_final(x)

        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens=50,
        temperature=1.0,
    ):

        self.eval()

        for _ in range(max_new_tokens):

            tokens = tokens[:, -CONTEXT_LENGTH:]

            logits = self(tokens)

            logits = logits[:, -1, :]
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            tokens = torch.cat(
                (tokens, next_token),
                dim=1
            )

        return tokens

    def count_parameters(self):
        """
        Returns the number of trainable parameters.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )