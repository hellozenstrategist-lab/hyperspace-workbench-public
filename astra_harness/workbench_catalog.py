"""Source-backed architecture descriptions for the local read-only workbench."""
from __future__ import annotations

import ast
from pathlib import Path


def _component(identifier, name, subtitle, category, source, description, inputs, outputs, invariants):
    return {
        "id": identifier, "name": name, "subtitle": subtitle, "category": category,
        "source": "astra_harness/" + source, "description": description,
        "inputs": inputs, "outputs": outputs, "invariants": invariants,
    }


def _defaults(root):
    defaults = {"research": {}, "limits": {}}
    root = Path(root).resolve()
    source = root / "astra_harness" / "research.py"
    try:
        if source.is_symlink() or source.parent.is_symlink() or not source.resolve().is_relative_to(root):
            return defaults
        if source.stat().st_size > 1_000_000:
            return defaults
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return defaults
    names = {"DEFAULT_MODELS": "research", "DEFAULT_LIMITS": "limits"}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                continue
            if isinstance(value, dict):
                defaults[names[target.id]] = {
                    key: item for key, item in value.items()
                    if isinstance(key, str) and isinstance(item, (str, int, float, bool))
                }
    return defaults


def get_catalog(root):
    components = [
        _component("cli", "Command line", "Choose a mode and saved run", "interface", "cli.py",
                   "Parses commands and selects the mission, benchmark, research, status, or replay workflow.",
                   ["Command arguments", "Explicit configuration", "Run directory"],
                   ["Coordinator or research invocation", "Local status output"],
                   ["Research and mission are separate workflows", "Reading saved status does not start models"]),
        _component("mission", "Mission contract", "Custom workers and final answers", "control", "mission.py",
                   "Extends the coordinator with custom worker prompts, publication receipts, and final-answer validation.",
                   ["Mission configuration", "Worker publications", "Final answers"],
                   ["Configured task graph", "Protocol report", "Declared evidence use"],
                   ["Receipt codes must match published findings", "Protocol completion is not general factual correctness"]),
        _component("coordinator", "Coordinator", "Deterministic orchestration", "control", "coordinator.py",
                   "Owns worker lifecycle, publication handling, optional routing, run evaluation, and durable exports.",
                   ["Run controls", "Model tool calls", "Worker lifecycle events"],
                   ["Concurrent worker turns", "Routing decisions", "Reports and snapshots"],
                   ["A run directory has one owner", "Completed replay adds no new model turns"]),
        _component("codex-runtime", "Codex runtime", "Persistent native threads", "runtime", "codex_runtime.py",
                   "Runs concurrent Codex app-server threads with native subscription authentication and persistent recovery records.",
                   ["Worker prompts", "Tool schemas", "Peer steering messages"],
                   ["Tool calls", "Final model messages", "Lifecycle and usage events"],
                   ["The requested model is checked", "Uncertain starts require reconciliation", "Worker overlap includes tool waits"]),
        _component("openrouter-runtime", "OpenRouter runtime", "Independent HTTP conversations", "runtime", "openrouter_runtime.py",
                   "Maintains separate worker conversations and durable request transcripts while executing registered host tools.",
                   ["Exact model ID", "Prompts and tool schemas", "Bounded request policy"],
                   ["Validated tool calls", "Submission results", "Request and usage records"],
                   ["Model substitution is rejected", "Incomplete usage stays unknown", "Client request overlap does not prove provider inference overlap"]),
        _component("knowledge-store", "Knowledge store", "SQLite is canonical", "storage", "knowledge_store.py",
                   "Persists runs, workers, knowledge events, routing decisions, receipts, tool results, and an audit ledger.",
                   ["Validated knowledge events", "Delivery state changes", "Evidence bytes"],
                   ["Canonical event records", "Durable receipts", "Snapshot and ledger exports"],
                   ["Publication and fanout are transactional", "Evidence is content addressed", "Acknowledgement and incorporation are separate states"]),
        _component("hyperspace-index", "HyperspaceDB", "Poincaré distance lookup", "storage", "hyperspace_backend.py",
                   "Indexes hierarchy-derived coordinates in the local Hyperspace service and returns distances for optional routing.",
                   ["Node IDs and coordinates", "Metadata", "Neighbor queries"],
                   ["Indexed neighbors and distances", "Persistent ID mappings"],
                   ["SQLite knowledge remains canonical", "Coordinates are derived from hierarchy, not learned embeddings", "Lost mapping collisions are rejected"]),
        _component("router", "Selective router", "Five routing modes", "routing", "router.py",
                   "Combines explicit dependencies, graph relations, lexical similarity, geometry, novelty, and context allowances.",
                   ["Knowledge event", "Worker attention state", "Index distances"],
                   ["Per-recipient delivery decisions", "Reasons and distance provenance"],
                   ["Required dependencies override optional filters", "Graph is an independent baseline", "Scores are heuristic, not calibrated probabilities"]),
        _component("attention", "Attention state", "Task anchors and budgets", "routing", "attention.py",
                   "Tracks worker tasks and hierarchy anchors used to bound optional context and support explicit reassignment.",
                   ["Worker task assignments", "Available task paths", "Attention policy"],
                   ["Worker anchors", "Assignment records", "Routing context"],
                   ["Reassignment is recorded explicitly", "Context allowance is not a provider token cap"]),
        _component("event-bus", "Peer event bus", "Deliver → read → acknowledge", "routing", "event_bus.py",
                   "Pumps durable peer deliveries, exposes bounded inbox reads, and validates explicit evidence-code acknowledgements.",
                   ["Pending delivery rows", "Worker inbox requests", "Receipt acknowledgements"],
                   ["Steering messages", "Read receipts", "Acknowledged delivery state"],
                   ["Queue acceptance is not model acknowledgement", "The exact code actually read is required", "Ambiguous sends are not blindly repeated"]),
        _component("evidence-audit", "Independent audit", "Reconstruct from saved evidence", "control", "evidence_audit.py",
                   "Checks saved manifests, ledger chains, evidence hashes, private-code receipts, and final incorporation independently of live execution.",
                   ["Manifest and fixture", "Evidence and ledger exports", "Final outputs"],
                   ["Acceptance checks", "Failure explanations", "Explicit evidence limits"],
                   ["Hashes establish local integrity, not external truth", "Fixture success does not certify arbitrary missions"]),
        _component("notifications", "Photon notifications", "Durable, bounded delivery", "interface", "photon_notifier.py",
                   "Applies notification policy, persistent deduplication, rate limits, and sidecar delivery handling.",
                   ["Run lifecycle events", "Verified-result notifications", "Notification policy"],
                   ["Outbox records", "Sidecar acknowledgements", "Quarantined uncertain sends"],
                   ["Ambiguous delivery is quarantined", "Sidecar acceptance is not a phone display or read receipt"]),
        _component("research-plan", "Research plan", "Head establishes fixed criteria", "research", "research.py",
                   "Creates worker assignments and fixed acceptance criteria, then incorporates the previous review in later rounds.",
                   ["Objective", "Supplied evidence", "Previous review"],
                   ["Structured criteria", "Worker assignments", "Plan artifact"],
                   ["Criteria cannot be weakened on later rounds", "Changed inputs require a new run", "Phase attempts retain immutable input hashes"]),
        _component("research-work", "Research workers", "Parallel work, staged review", "research", "research.py",
                   "Runs one to three worker nodes against assigned artifacts; configured browser collection can precede a separate falsification phase.",
                   ["Plan and assignment", "Available artifacts", "Configured observation tools"],
                   ["Findings with evidence IDs", "Unresolved questions", "Worker artifacts"],
                   ["Cited artifacts must be read before submission", "Worker completion cannot accept the whole project", "Research does not route through HyperspaceDB"]),
        _component("research-synthesis", "Synthesis", "Head assembles the candidate", "research", "research.py",
                   "Reads the current worker results and submits a candidate answer with its supporting artifacts and unresolved requirements.",
                   ["Every current worker result", "Objective and fixed criteria"],
                   ["Candidate answer", "Evidence references", "Unresolved requirements"],
                   ["A candidate is not accepted completion", "Validated structured submission completes the phase"]),
        _component("research-qc", "Independent QC", "Accept or request another round", "research", "research.py",
                   "Reviews the objective, every fixed criterion, worker evidence, and synthesized answer before accepting or returning actionable gaps.",
                   ["Candidate and worker artifacts", "Original objective", "Fixed criteria"],
                   ["Per-criterion checks", "Accept, revise, or blocked verdict", "Next-round feedback"],
                   ["Acceptance requires supported passing checks and no next steps", "Revise and actionable blocked verdicts continue within host limits", "Model approval does not independently establish truth"]),
        _component("research-artifacts", "Research artifacts", "Files and immutable attempts", "storage", "research.py",
                   "Stores content-addressed artifacts, research checkpoints, selected phase outputs, and separate numbered provider ledgers.",
                   ["Supplied evidence", "Structured phase submissions", "Runtime transcripts"],
                   ["research.json checkpoint", "research-report.json", "Phase attempt history"],
                   ["This store is separate from the mission SQLite bus", "Completed phases are reused on resume", "Uncertain requests remain in their original ledger"]),
        _component("browser-adapter", "Browser adapter", "Explicit, isolated observations", "interface", "browser_cdp.py",
                   "Connects only explicitly configured browser sessions and records bounded observations under the saved session policy.",
                   ["Explicit browser configuration", "Worker session ownership", "Allowed observation requests"],
                   ["Redacted observation artifacts", "Durable action intents", "Non-evidence for unavailable observations"],
                   ["A prompt alone cannot grant browser access", "Session ownership is enforced", "Uncertain actions are not treated as evidence"]),
    ]
    mission_nodes = [
        ("cli", 40, 65), ("mission", 280, 65), ("coordinator", 520, 65),
        ("codex-runtime", 790, 25), ("openrouter-runtime", 1040, 135),
        ("attention", 40, 300), ("router", 280, 300), ("knowledge-store", 520, 300),
        ("event-bus", 790, 300), ("hyperspace-index", 280, 550),
        ("evidence-audit", 520, 550), ("notifications", 790, 550),
    ]
    mission_edges = [
        ("cli", "mission", "custom configuration", "control"),
        ("mission", "coordinator", "extends lifecycle", "control"),
        ("coordinator", "codex-runtime", "native worker turns", "control"),
        ("coordinator", "openrouter-runtime", "alternative provider", "control"),
        ("coordinator", "knowledge-store", "persist publications", "data"),
        ("attention", "router", "task anchors", "data"),
        ("knowledge-store", "router", "events and dependencies", "data"),
        ("knowledge-store", "hyperspace-index", "index coordinates", "data"),
        ("hyperspace-index", "router", "optional distances", "data"),
        ("router", "event-bus", "selected deliveries", "data"),
        ("event-bus", "codex-runtime", "peer steering", "feedback"),
        ("event-bus", "openrouter-runtime", "queued peer context", "feedback"),
        ("event-bus", "knowledge-store", "read and ACK receipts", "feedback"),
        ("knowledge-store", "evidence-audit", "saved evidence", "data"),
        ("coordinator", "notifications", "lifecycle events", "control"),
    ]
    research_nodes = [
        ("cli", 40, 95), ("research-plan", 300, 95), ("research-work", 570, 95),
        ("research-synthesis", 850, 95), ("research-qc", 1050, 335),
        ("research-artifacts", 570, 355), ("browser-adapter", 300, 355),
        ("openrouter-runtime", 570, 565),
    ]
    research_edges = [
        ("cli", "research-plan", "objective and controls", "control"),
        ("research-plan", "research-work", "fixed criteria and tasks", "control"),
        ("research-work", "research-synthesis", "current worker results", "data"),
        ("research-synthesis", "research-qc", "candidate answer", "data"),
        ("research-qc", "research-plan", "revise within limits", "feedback"),
        ("browser-adapter", "research-work", "configured observations", "data"),
        ("research-work", "research-artifacts", "validated findings", "data"),
        ("research-artifacts", "research-synthesis", "read cited evidence", "data"),
        ("research-artifacts", "research-qc", "evidence and history", "data"),
        ("openrouter-runtime", "research-artifacts", "immutable phase ledgers", "data"),
    ]

    def scene(identifier, name, description, nodes, edges):
        return {
            "id": identifier, "name": name, "description": description,
            "nodes": [{"id": node_id, "x": position_x, "y": position_y}
                      for node_id, position_x, position_y in nodes],
            "edges": [{"source": source, "target": target, "label": label, "kind": kind}
                      for source, target, label, kind in edges],
        }

    return {
        "components": components,
        "scenes": [
            scene("mission", "Mission harness", "Durable peer evidence sharing with selective optional routing.",
                  mission_nodes, mission_edges),
            scene("research", "Research review loop", "Artifact-based planning, worker analysis, synthesis, and independent review.",
                  research_nodes, research_edges),
        ],
        "defaults": _defaults(root),
    }
