"""Versioned, bounded knowledge events; no provider credentials belong here."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid

AGENTS = ("agent-a", "agent-b", "agent-c")
ALL_AGENTS = tuple("agent-" + letter for letter in "abcde")


def worker_ids(count=3):
    """Canonical bounded worker identities; old three-worker runs keep their IDs."""
    if type(count) is not int or not 2 <= count <= len(ALL_AGENTS):
        raise ValueError("Worker count must be an integer from 2 to 5")
    return ALL_AGENTS[:count]


KINDS = {"claim", "evidence", "hypothesis", "contradiction", "task", "result", "scope_change"}
VERIFICATIONS = {"unverified", "observed", "reproduced", "disproved"}
PRIORITIES = {"routine": 2, "important": 1, "critical": 0}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(value: bytes):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class KnowledgeEvent:
    event_id: str
    run_id: str
    author_agent: str
    type: str
    claim: str
    evidence_refs: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    verification_status: str = "unverified"
    priority: str = "routine"
    requested_action: str | None = None
    created_at: str = field(default_factory=now)
    schema_version: int = 1
    scope_path: list[str] = field(default_factory=lambda: ["root"])
    dependencies: list[str] = field(default_factory=list)
    revision_of: str | None = None
    central: bool = False
    safety_constraint: bool = False

    def validate(self):
        uuid.UUID(self.event_id)
        uuid.UUID(self.run_id)
        if self.schema_version != 1:
            raise ValueError("Unsupported knowledge schema version")
        if self.author_agent not in (*ALL_AGENTS, "coordinator"):
            raise ValueError("Unknown author")
        if self.type not in KINDS or self.verification_status not in VERIFICATIONS or self.priority not in PRIORITIES:
            raise ValueError("Invalid event enum")
        if not isinstance(self.claim, str) or not 1 <= len(self.claim) <= 2000:
            raise ValueError("claim must contain 1..2000 characters")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite in [0,1]")
        timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("created_at needs a timezone")
        for name in ("evidence_refs", "parent_ids", "contradicts", "dependencies", "scope_path"):
            items = getattr(self, name)
            if not isinstance(items, list) or len(items) > 32 or any(not isinstance(x, str) or not 1 <= len(x) <= 200 for x in items):
                raise ValueError(f"Invalid bounded list {name}")
            if len(items) != len(set(items)) and name != "scope_path":
                raise ValueError(f"Duplicate {name}")
        if not self.scope_path or any(x not in ALL_AGENTS for x in self.dependencies):
            raise ValueError("Invalid scope/dependency")
        if any(len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in self.evidence_refs):
            raise ValueError("Evidence references must be SHA-256 digests")
        if self.requested_action is not None and (not isinstance(self.requested_action, str) or len(self.requested_action) > 500):
            raise ValueError("Requested action exceeds bound")
        if not isinstance(self.central, bool) or not isinstance(self.safety_constraint, bool):
            raise ValueError("Scope flags must be booleans")
        if self.revision_of is not None and (not isinstance(self.revision_of, str) or not 1 <= len(self.revision_of) <= 200):
            raise ValueError("Invalid revision reference")
        if len(canonical(asdict(self)).encode()) > 16000:
            raise ValueError("Event exceeds byte limit")
        return self

    def to_dict(self):
        return asdict(self.validate())

    @classmethod
    def from_dict(cls, value):
        return cls(**value).validate()

    def compact(self):
        return {k: v for k, v in self.to_dict().items() if k in {
            "schema_version", "event_id", "author_agent", "type", "claim", "evidence_refs",
            "confidence", "verification_status", "priority", "requested_action", "scope_path"}}


def atomic_json(path, value):
    """Durable JSON replacement with private permissions and directory fsync."""
    import os
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
