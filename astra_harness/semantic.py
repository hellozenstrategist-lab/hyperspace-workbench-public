"""Deterministic local lexical hashing; these vectors are NOT learned semantics."""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

ENCODER_VERSION = "lexical-blake2b-v1"
DEFAULT_DIMENSIONS = 256


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).casefold(), flags=re.UNICODE)


def embed(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 8 or dimensions > 65536:
        raise ValueError("dimensions must be an integer from 8 to 65536")
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    features = [("w:" + token, 1.0) for token in tokens]
    features += [("b:" + left + "\0" + right, 0.5) for left, right in zip(tokens, tokens[1:])]
    for feature, weight in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=16, person=b"astra-lex-v1").digest()
        bucket = int.from_bytes(digest[:8], "big") % dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[bucket] += sign * weight
    norm = math.sqrt(math.fsum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("vectors must be finite")
    denominator = math.sqrt(math.fsum(v*v for v in left) * math.fsum(v*v for v in right))
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(left, right)) / denominator))


def similarity(left_text: str, right_text: str, dimensions: int = DEFAULT_DIMENSIONS) -> float:
    return cosine(embed(left_text, dimensions), embed(right_text, dimensions))


@dataclass(frozen=True)
class HashedTokenEncoder:
    dimensions: int = DEFAULT_DIMENSIONS

    def encode(self, text: str) -> list[float]:
        return embed(text, self.dimensions)

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]

    def describe(self) -> dict:
        return {"kind": "lexical_hashing", "version": ENCODER_VERSION, "dimensions": self.dimensions, "learned": False, "remote_calls": False}


hash_vector = embed
lexical_embedding = embed
