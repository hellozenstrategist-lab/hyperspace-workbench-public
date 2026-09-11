# Hierarchy coordinates and attention routing

The harness keeps two different representations. `hyperbolic_index.position(path)` maps the canonical primary-parent path into a two-dimensional Poincare ball. `semantic.embed(text)` hashes local words and adjacent word pairs into a normalized 256-dimensional lexical vector. The second representation is deterministic lexical matching, **not a learned semantic embedding**. Neither representation calls a model, downloads weights, or invokes a remote embedding service. Text changes never silently move a hierarchy coordinate.

## Geometry contract

```python
from astra_harness.hyperbolic_index import position, distance, nearest

p = position(["root", "project", "document-organizer", "accuracy"])
d = distance(p, position(["root", "project", "document-organizer"]))
neighbors = nearest(p, {"latency": position(["root", "project", "document-organizer", "latency"])}, k=1)
```

`path-sector-v1` assigns root depth zero and radius `tanh(0.65 * depth / 2)`. Therefore the Poincare distance from the origin is exactly `0.65 * depth`, up to floating-point rounding. Length-prefixed hashes of immutable path identifiers determine angular offsets; each successive level uses 30% of the preceding angular scale. Existing siblings do not move when another child is added. Paths support up to 16 edges from the root. Coordinates must be finite, have exactly two components, and lie strictly inside the unit ball.

The layout is a practical, deterministic **structural heuristic**. It has no guarantee that all siblings occupy disjoint sectors, no bounded distortion guarantee, and no learned understanding of a claim. Hash collisions and crowded sectors can put unrelated branches near one another. A cross-branch relationship is an explicit graph edge; it does not become correct because two coordinates happen to be near. All paths should use one canonical root and stable node identifiers; an empty path and a one-node root both map to the origin.

Exact distance is `acosh(1 + 2 ||x-y||² / ((1-||x||²)(1-||y||²)))`. The local `nearest` implementation is a deterministic exhaustive reference. Production queries use the actual upstream HyperspaceDB Poincare index. It is approximate at scale; small-fixture distance/rank agreement does not prove recall at larger scales.

Curvature is fixed at **-1**, with a unit-radius ball and Python double-precision arithmetic. The origin is `[0,0]`; there is no angular direction to normalize there. Generated depths are bounded so they remain away from the singular boundary. Validation rejects nonfinite values, boolean coordinates and squared norms at or above `1-1e-9`. External points are not silently clamped into the ball: callers must make any projection an explicit coordinate revision. The metric returns exact zero for identical validated inputs, uses compensated sums, and clamps the `acosh` argument to at least1 against roundoff. Extremely close distinct points can round to zero in this version, so sub-machine-precision distance should never establish semantic identity or deduplication.

## Updates and persistence

`coordinate_record(path)` returns `{path, position, layout_version, revision, strategy}`. `update_coordinate(record, new_path)` increments the coordinate revision when the primary-parent path changes. Semantic text edits or extra relationship edges leave the coordinate revision unchanged. `serialize`/`deserialize` validate the stored path, version and coordinates; unknown versions fail and require explicit migration.

When a graph node is reparented, its descendants' primary paths must also be updated. The canonical store must commit new paths, coordinate versions/revisions, and projection-outbox entries together. Projection writes can then be retried by logical node ID. Ordered revision processing belongs to that outbox; HyperspaceDB is a rebuildable projection and does not make a distributed transaction with SQLite. A later layout version needs an explicit canonical migration followed by reindexing. Do not mix coordinate versions in one search collection.

Keep the backend's stable node-ID mapping SQLite file with its associated collection. Sharing a collection requires sharing the same mapping file. The backend checks an existing vector ID's logical identity before overwriting it and fails on a mapping conflict. If the mapping is lost, restore it or rebuild a new collection from the canonical graph. Do not connect independent allocators to the same collection concurrently.

## Routing contract

`Router.route(event, agents, owners, dependencies, mode, neighbor_distances)` returns one auditable decision per agent. Native KnowledgeEvent fields are read directly: `author_agent`, `claim`, `scope_path`, `parent_ids`, `contradicts`, `dependencies`, `priority`, `verification_status`, `central`, and `safety_constraint`.

Every mode first delivers explicit dependencies. Safety constraints, scope changes, central knowledge, disproved claims, and verified critical findings (observed/reproduced) broadcast to the other workers. Contradictions reach affected claim owners. These deliveries override radius, lexical score, novelty and attention budget. A sender does not receive its own event.

| Mode | Optional delivery |
| --- | --- |
| broadcast | Every peer |
| flat | Local lexical cosine at least 0.20 |
| graph | Ancestor/descendant, immediate sibling task, or owner of a referenced parent node |
| hyperbolic | Poincare distance within the recipient's radius, default 1.5 |
| hybrid | Within radius and either graph-related or lexical cosine at least 0.20 |

Already-seen events, zero novelty, and exhausted per-agent budgets suppress optional attention. An explicitly enabled `diversity_probe` admits at most one additional eligible peer to expose another branch to competing evidence. Its inclusion is recorded. Novelty and budget inputs are supplied by the canonical coordinator; they are not inferred by a hidden model. Scores are ranking heuristics, labelled `score_calibrated: false`; they are not probabilities of relevance or factual confidence.

The coordinator indexes the three worker anchors with `kind=anchor`, `agent`, and `run_id`, searches the real server using those filters, and supplies the returned distances by agent ID. Decisions report `distance_source=hyperspace_index`. Missing distances fall back to the local exact metric in the library; acceptance runs should require all three actual server distances and fail if any are missing. Explicit reason strings such as `within_attention_radius`, `outside_attention_radius`, `parent_graph_owner`, `explicit_dependency`, and `critical_broadcast` explain each decision.

## Evidence and limits

`tests/fixtures/wrong_branch.json` is deliberately adversarial: an accuracy task in the correct project uses different words while a marketing branch repeats the query words. It demonstrates a case where known branch structure corrects lexical distraction. It does not show that geometry universally improves relevance; the graph baseline also succeeds on this fixture. Required acceptance messages are explicitly dependent and therefore reach all peers in all five modes. Measure optional branch probes separately when comparing selectivity.

Property tests check distance symmetry, identity, triangle inequality, radial depth, interior bounds, normalization and versioned reparenting. The actual-server check verifies filters, exact distances on a small fixture, nearest-neighbor routing, same-ID updates, record readback, idempotent reconciliation, mapping conflict protection, and a graceful server restart with persisted vectors.

`scripts/benchmark_geometry.py --endpoint <current-private-endpoint> --output artifacts/geometry_benchmark.json` reproduces all five routing modes against the real index without model calls or a restart. On this one constructed fixture, broadcast reaches the relevant peer and the distractor, flat lexical reaches only the distractor, and graph/hyperbolic/hybrid reach only the relevant peer. Its separate dependency check confirms that all modes deliver both mandatory peers. The JSON records actual neighbor distances, decisions, the single-fixture scope, and the absence of a general quality claim.
