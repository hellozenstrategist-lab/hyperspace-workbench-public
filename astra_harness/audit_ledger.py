"""Append-only hash chain, written in the same transaction as durable state.

Detects accidental alteration. A local administrator can rewrite the entire
chain; export and externally anchor head hashes for stronger tamper evidence.
"""
from __future__ import annotations
import hashlib
from .schema import canonical, now


def append(conn, run_id, kind, data, *, key=None):
    if key:
        found = conn.execute("SELECT seq FROM ledger WHERE run_id=? AND idem_key=?", (run_id, key)).fetchone()
        if found:
            return found[0]
    previous = conn.execute("SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
    prev = previous[0] if previous else "0" * 64
    timestamp = now()
    body = {"run_id": run_id, "kind": kind, "at": timestamp, "data": data, "prev_hash": prev, "idem_key": key}
    checksum = hashlib.sha256(canonical(body).encode()).hexdigest()
    cur = conn.execute("INSERT INTO ledger(run_id,kind,at,data,prev_hash,hash,idem_key) VALUES(?,?,?,?,?,?,?)",
                       (run_id, kind, timestamp, canonical(data), prev, checksum, key))
    return cur.lastrowid


def verify(rows):
    prev = "0" * 64
    failures = []
    import json
    for row in rows:
        value = dict(row)
        body = {k: value[k] for k in ("run_id", "kind", "at", "prev_hash", "idem_key")}
        body["data"] = json.loads(value["data"]) if isinstance(value["data"], str) else value["data"]
        expected = hashlib.sha256(canonical(body).encode()).hexdigest()
        if value["prev_hash"] != prev or value["hash"] != expected:
            failures.append(value["seq"])
        prev = value["hash"]
    return {"valid": not failures, "invalid_sequences": failures, "head": prev, "entries": len(rows)}
