"""Versioned, deterministic hierarchy layout and exact Poincare distances.

This is a structural coordinate layout, not a learned semantic embedding. A path
is the primary-parent chain. Other graph edges are kept by the canonical graph.
"""
from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

LAYOUT_VERSION = "path-sector-v1"
MAX_DEPTH = 16
RADIAL_STEP = 0.65
BRANCH_SHRINK = 0.30


def normalize_path(path: Sequence[str]) -> list[str]:
    if isinstance(path, (str, bytes)):
        raise TypeError("path must be a sequence of node identifiers, not a string")
    result = []
    for item in path:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("path identifiers must be nonempty strings")
        result.append(unicodedata.normalize("NFC", item.strip()))
    if len(result) > MAX_DEPTH + 1:
        raise ValueError(f"layout supports at most {MAX_DEPTH} edges from its root")
    return result


def _unit_hash(path: Sequence[str]) -> float:
    # Length prefixes avoid ambiguous paths such as ['ab','c'] and ['a','bc'].
    payload = b"".join(len(part.encode()).to_bytes(4, "big") + part.encode() for part in path)
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"astra-path-v1").digest(), "big")
    return value / 2**64


def validate_position(value: Sequence[float]) -> list[float]:
    if len(value) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        raise ValueError("position must contain two finite numbers")
    point = [float(v) for v in value]
    if sum(v * v for v in point) >= 1 - 1e-9:
        raise ValueError("position must be strictly inside the Poincare unit ball")
    return point


def position(path: Sequence[str], semantic_text: str = "", related_positions: Iterable[Sequence[float]] = ()) -> list[float]:
    """Automatically derive coordinates solely from a canonical primary path.

    Root/empty paths are at the origin. Radius encodes depth exactly through
    r=tanh(depth*RADIAL_STEP/2). Prefixes share progressively narrower angular
    neighborhoods; identifier hashing fixes sibling directions without moving
    existing siblings when a new child is added. Distortion is not bounded.

    semantic_text and related_positions are accepted for the public contract,
    but intentionally do not warp a hierarchy coordinate when facts/edges change.
    """
    del semantic_text, related_positions
    parts = normalize_path(path)
    depth = max(0, len(parts) - 1)
    if depth == 0:
        return [0.0, 0.0]
    angle = 2 * math.pi * _unit_hash(parts[:2])
    for level in range(2, len(parts)):
        angle += (_unit_hash(parts[:level + 1]) - 0.5) * 2 * math.pi * BRANCH_SHRINK ** (level - 1)
    radius = math.tanh(depth * RADIAL_STEP / 2)
    return validate_position([radius * math.cos(angle), radius * math.sin(angle)])


def distance(x: Sequence[float], y: Sequence[float]) -> float:
    left, right = validate_position(x), validate_position(y)
    if left == right:
        return 0.0
    delta = math.fsum((a - b) ** 2 for a, b in zip(left, right))
    denominator = (1 - math.fsum(a * a for a in left)) * (1 - math.fsum(b * b for b in right))
    return math.acosh(max(1.0, 1.0 + 2.0 * delta / denominator))


def nearest(query: Sequence[float], items: Mapping[str, Sequence[float]], k: int = 5) -> list[dict]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    validate_position(query)
    ranked = [{"id": ident, "distance": distance(query, point), "position": list(point)} for ident, point in items.items()]
    return sorted(ranked, key=lambda item: (item["distance"], str(item["id"])))[:k]


@dataclass(frozen=True)
class CoordinateRecord:
    path: tuple[str, ...]
    position: tuple[float, float]
    layout_version: str = LAYOUT_VERSION
    revision: int = 1
    strategy: str = "primary-path"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["path"] = list(self.path)
        value["position"] = list(self.position)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "CoordinateRecord":
        if value.get("layout_version") != LAYOUT_VERSION:
            raise ValueError("unsupported layout version; explicit migration is required")
        if value.get("strategy", "primary-path") != "primary-path":
            raise ValueError("unsupported coordinate strategy")
        path = normalize_path(value["path"])
        point = validate_position(value["position"])
        if any(abs(a - b) > 1e-12 for a, b in zip(point, position(path))):
            raise ValueError("stored coordinate does not match its recorded path/version")
        revision = value.get("revision", 1)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ValueError("coordinate revision must be a positive integer")
        return cls(tuple(path), tuple(point), revision=revision)


def coordinate_record(path: Sequence[str], revision: int = 1) -> dict:
    parts = normalize_path(path)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("coordinate revision must be a positive integer")
    return CoordinateRecord(tuple(parts), tuple(position(parts)), revision=revision).to_dict()


def update_coordinate(previous: dict, path: Sequence[str], *, semantic_text: str = "", related_positions: Iterable[Sequence[float]] = ()) -> dict:
    """Reparent/path changes create a new coordinate revision; text/links do not.

    The graph owner must update paths for every descendant of a moved node and
    persist canonical path/coordinate/revision plus its projection-outbox entry
    in the same transaction. Stable immutable node IDs should be used in paths.
    """
    old = CoordinateRecord.from_dict(previous)
    parts = normalize_path(path)
    del semantic_text, related_positions
    return coordinate_record(parts, old.revision + (tuple(parts) != old.path))


def serialize(record: dict) -> dict:
    return CoordinateRecord.from_dict(record).to_dict()


def deserialize(value: dict) -> dict:
    return CoordinateRecord.from_dict(value).to_dict()
