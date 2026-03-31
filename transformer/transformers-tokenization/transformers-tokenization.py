import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        """
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        # Add special tokens
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        next_id = len(special_tokens)

        # Add words from texts
        for text in texts:
            for word in text.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = next_id
                    self.id_to_word[next_id] = word
                    next_id += 1

        self.vocab_size = next_id

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        """
        tokens = text.split()
        unk_id = self.word_to_id[self.unk_token]

        ids = []
        for word in tokens:
            ids.append(self.word_to_id.get(word, unk_id))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for i in ids:
            words.append(self.id_to_word.get(i, self.unk_token))

        return " ".join(words)