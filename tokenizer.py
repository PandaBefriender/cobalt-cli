"""
tokenizer.py

TinyGPT Word-Level Tokenizer

Responsibilities:
- Build vocabulary from text
- Encode text into token IDs
- Decode token IDs back into text
- Save/load vocabulary
"""

import json
import re
from collections import Counter

from config import MIN_FREQUENCY


class Tokenizer:

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):

        self.word_to_id = {
            self.PAD: 0,
            self.UNK: 1,
        }

        self.id_to_word = {
            0: self.PAD,
            1: self.UNK,
        }


    # ======================================================
    # Text Cleaning
    # ======================================================

    def tokenize(self, text):

        """
        Converts raw text into clean tokens.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )


    # ======================================================
    # Build Vocabulary
    # ======================================================

    def train(self, text):

        words = self.tokenize(text)

        counts = Counter(words)

        next_id = 2

        for word, count in counts.items():

            if count >= MIN_FREQUENCY:

                self.word_to_id[word] = next_id

                self.id_to_word[next_id] = word

                next_id += 1



    # ======================================================
    # Encode Text
    # ======================================================

    def encode(self, text):

        words = self.tokenize(text)

        ids = []

        for word in words:

            token_id = self.word_to_id.get(
                word,
                self.word_to_id[self.UNK]
            )

            ids.append(token_id)

        return ids



    # ======================================================
    # Decode Tokens
    # ======================================================

    def decode(self, ids):

        words = []

        for token_id in ids:

            word = self.id_to_word.get(
                int(token_id),
                self.UNK
            )

            words.append(word)

        return " ".join(words)



    # ======================================================
    # Save Vocabulary
    # ======================================================

    def save(self, path="vocab.json"):

        with open(path, "w") as file:

            json.dump(
                self.word_to_id,
                file,
                indent=4
            )



    # ======================================================
    # Load Vocabulary
    # ======================================================

    def load(self, path="vocab.json"):

        with open(path, "r") as file:

            self.word_to_id = json.load(file)


        self.id_to_word = {
            int(index): word
            for word, index in self.word_to_id.items()
        }



    # ======================================================
    # Vocabulary Size
    # ======================================================

    @property
    def vocab_size(self):

        return len(self.word_to_id)



# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    sample = """
    The sky is blue.
    The sky is bright!
    The sun is bright.
    """


    tokenizer = Tokenizer()

    tokenizer.train(sample)


    print(tokenizer.word_to_id)


    encoded = tokenizer.encode(
        "The sky is blue!"
    )

    print(encoded)


    decoded = tokenizer.decode(
        encoded
    )

    print(decoded)