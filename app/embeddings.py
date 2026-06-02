from __future__ import annotations

import hashlib
import re

import numpy as np


class HashingEmbedder:
    """Deterministic local embeddings for FAISS without external secrets."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, bucket] += sign

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _tokens(self, text: str) -> list[str]:
        words = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams
