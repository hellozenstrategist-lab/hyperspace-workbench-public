"""Synchronous local HyperspaceDB projection through its real Python SDK RPCs.

The canonical graph/outbox owns correctness. This is a rebuildable vector index,
not an authoritative event store. No SDK text/vectorize path is invoked.
"""
from __future__ import annotations

import ipaddress
import json
import math
import re
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .hyperbolic_index import LAYOUT_VERSION, validate_position


def _local_host(host):
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("endpoint must use localhost or a private numeric address") from exc
    if not (address.is_private or address.is_loopback) or address.is_unspecified:
        raise ValueError("this backend is restricted to an explicit local/private service")


class HyperspaceBackend:
    def __init__(self, endpoint: str, *, state_path, user_id="astraharness", collection="knowledge", http_endpoint=None, timeout=5.0):
        parsed = urllib.parse.urlsplit("http://" + endpoint)
        if not parsed.hostname or not parsed.port or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("endpoint must be host:port")
        _local_host(parsed.hostname)
        for value in (user_id, collection):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
                raise ValueError("identity/collection must contain 1..100 letters, digits, underscores or hyphens")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        host = "[" + parsed.hostname + "]" if ":" in parsed.hostname else parsed.hostname
        self.http_endpoint = (http_endpoint or f"http://{host}:50050").rstrip("/")
        http = urllib.parse.urlsplit(self.http_endpoint)
        if http.scheme != "http" or http.username or http.password or http.path or http.query or http.fragment or not http.port:
            raise ValueError("http_endpoint must be an explicit local http://host:port")
        _local_host(http.hostname)
        if http.hostname != parsed.hostname:
            raise ValueError("HTTP and gRPC endpoints must address the same local host")
        # Import before opening persistent state so a missing SDK is a clean
        # setup error, without an abandoned database connection.
        import grpc
        from hyperspace.proto import hyperspace_pb2, hyperspace_pb2_grpc
        self.endpoint, self.user_id, self.collection, self.timeout = endpoint, user_id, collection, timeout
        self._lock = threading.RLock()
        state_path = Path(state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(state_path, check_same_thread=False, timeout=10)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE IF NOT EXISTS ids (namespace TEXT NOT NULL, node_id TEXT NOT NULL, vector_id INTEGER NOT NULL CHECK(vector_id>0 AND vector_id<=4294967295), PRIMARY KEY(namespace,node_id), UNIQUE(namespace,vector_id))")
        self._db.commit()
        self._namespace = user_id + ":" + collection
        # Generated stubs are shipped by the upstream Python SDK. Explicit
        # channels add per-request deadlines and disable inherited HTTP proxies.
        self._proto = hyperspace_pb2
        self._channel = grpc.insecure_channel(endpoint, options=(("grpc.enable_http_proxy", 0), ("grpc.max_receive_message_length", 4 * 1024 * 1024)))
        self._stub = hyperspace_pb2_grpc.DatabaseStub(self._channel)
        self._metadata = (("x-hyperspace-user-id", user_id),)
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            if self._rpc("HealthCheck", self._proto.Empty()).status != "SERVING":
                raise RuntimeError("HyperspaceDB is not serving")
            self._ensure_collection()
        except BaseException:
            self.close()
            raise

    def _rpc(self, method, request):
        return getattr(self._stub, method)(request, metadata=self._metadata, timeout=self.timeout)

    def _http_json(self, path, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(self.http_endpoint + path, data=data, headers={"Content-Type": "application/json", "x-hyperspace-user-id": self.user_id}, method="GET" if body is None else "POST")
        with self._http.open(request, timeout=self.timeout) as response:
            raw = response.read(1024 * 1024)
            return json.loads(raw) if raw else {}

    def _ensure_collection(self):
        names = {item["name"] for item in self._http_json("/api/collections")}
        if self.collection not in names:
            self._http_json("/api/collections", {"name": self.collection, "dimension": 2, "metric": "poincare", "quantization": "none"})
        stats = self._http_json("/api/collections/" + self.collection + "/stats")
        if stats.get("dimension") != 2 or str(stats.get("metric", "")).lower() != "poincare" or str(stats.get("quantization", "")).lower() not in ("none", "f64", "float64"):
            raise RuntimeError(f"collection must be 2D Poincare with quantization none: {stats}")

    def _vector_id(self, node_id, create=False):
        if not isinstance(node_id, str) or not node_id or len(node_id) > 512:
            raise ValueError("node_id must be a nonempty string of at most 512 characters")
        with self._lock:
            # IMMEDIATE serializes independent backend instances/processes using
            # the same mapping file. A failed remote write keeps its stable ID.
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT vector_id FROM ids WHERE namespace=? AND node_id=?", (self._namespace, node_id)).fetchone()
                if row:
                    result = row[0]
                elif create:
                    result = self._db.execute("SELECT COALESCE(MAX(vector_id),0)+1 FROM ids WHERE namespace=?", (self._namespace,)).fetchone()[0]
                    self._db.execute("INSERT INTO ids VALUES (?,?,?)", (self._namespace, node_id, result))
                else:
                    result = None
                self._db.commit()
                return result
            except BaseException:
                self._db.rollback()
                raise

    def upsert(self, node_id: str, position, metadata: dict | None = None) -> dict:
        point = validate_position(position)
        metadata = dict(metadata or {})
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 65536:
            raise ValueError("projection metadata exceeds 64 KiB")
        vector_id = self._vector_id(node_id, create=True)
        existing = self._rpc("GetPoints", self._proto.GetPointsRequest(collection=self.collection, ids=[vector_id])).points
        if existing and existing[0].metadata.get("astra_node_id") != node_id:
            raise RuntimeError("projection ID mapping disagrees with persistent server; restore its mapping or rebuild a new collection from the canonical graph")
        # Scalar fields remain independently filterable in the actual index.
        fields = {key: str(value) for key, value in metadata.items() if isinstance(value, (str, int, float, bool)) and not key.startswith("astra_")}
        fields.update(astra_node_id=node_id, astra_record=encoded, astra_layout_version=LAYOUT_VERSION)
        response = self._rpc("Insert", self._proto.InsertRequest(collection=self.collection, id=vector_id, vector=point, metadata=fields, durability=self._proto.STRICT))
        if not response.success:
            raise RuntimeError("HyperspaceDB rejected the projection upsert")
        return {"id": node_id, "vector_id": vector_id, "stored": True, "durability": "strict", "layout_version": LAYOUT_VERSION}

    @staticmethod
    def _decode(row):
        fields = dict(row.metadata)
        if "astra_node_id" not in fields or "astra_record" not in fields:
            raise RuntimeError("collection contains a point outside this harness projection")
        result = {"id": fields["astra_node_id"], "vector_id": row.id, "metadata": json.loads(fields["astra_record"])}
        if hasattr(row, "distance"):
            result["distance"] = row.distance
        if hasattr(row, "vector"):
            result["position"] = list(row.vector)
        return result

    def search(self, vector, k=5, filter=None) -> list[dict]:
        point = validate_position(vector)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10000:
            raise ValueError("k must be an integer in 1..10000")
        filters = {str(key): str(value) for key, value in (filter or {}).items()}
        response = self._rpc("Search", self._proto.SearchRequest(collection=self.collection, vector=point, top_k=k, filter=filters))
        return [self._decode(row) for row in response.results]

    def get(self, node_id) -> dict | None:
        vector_id = self._vector_id(node_id)
        if vector_id is None:
            return None
        rows = self._rpc("GetPoints", self._proto.GetPointsRequest(collection=self.collection, ids=[vector_id])).points
        return self._decode(rows[0]) if rows else None

    def health(self) -> dict:
        status = self._rpc("HealthCheck", self._proto.Empty()).status
        stats = self._http_json("/api/collections/" + self.collection + "/stats")
        return {"status": status, "endpoint": self.endpoint, "collection": self.collection, "user_id": self.user_id, "metric": "poincare", "dimensions": 2, "quantization": "none", "layout_version": LAYOUT_VERSION, "stats": stats}

    def reconcile(self, records) -> dict:
        count = 0
        for record in records:
            self.upsert(record.get("id", record.get("node_id")), record["position"], record.get("metadata", {}))
            count += 1
        return {"reapplied": count, "strategy": "idempotent_canonical_replay"}

    def close(self):
        if hasattr(self, "_channel"):
            self._channel.close()
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
