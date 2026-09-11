"""Concurrent knowledge workers with Codex or OpenRouter and shared HyperspaceDB."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .schema import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def endpoint(value):
    if value == "local":
        return None
    if value != "auto":
        return value
    output = subprocess.run([sys.executable, str(ROOT / "scripts/hyperspace_service.py"), "start"],
                            capture_output=True, text=True, timeout=60, check=True)
    return json.loads(output.stdout)["endpoint"]


def run_options(args, directory=None):
    from .runtime_factory import runtime_options
    from .schema import worker_ids
    saved = {}
    if directory and (directory / "manifest.json").exists():
        saved = json.loads((directory / "manifest.json").read_text())
    count = getattr(args, "workers", None) or saved.get("worker_count", 5)
    worker_ids(count)
    provider, model = runtime_options(args.provider or saved.get("provider", "codex"),
        args.model or saved.get("requested_model") or saved.get("model"))
    return {"worker_count": count, "provider": provider, "model": model}


def notifier_for(args, directory):
    if getattr(args, "photon", "off") == "off":
        return None
    from .photon_notifier import PhotonNotifier, NotificationPolicy
    # Acceptance explicitly requests a three-message test. Ordinary production
    # notifications retain quiet hours and limits in NotificationPolicy defaults.
    policy = (NotificationPolicy() if getattr(args, "command", None) in {"mission", "notifications"} else
              NotificationPolicy(quiet_start_hour=None, quiet_end_hour=None, min_interval_seconds=1))
    if getattr(args, "digest_interval", None) is not None:
        policy = replace(policy, digest_enabled=True)
    return PhotonNotifier(ROOT / "data/notifications.sqlite", base_url=args.photon_url,
                          state_path=args.photon_state, policy=policy, provider_idempotency=False)


def count_effects(directory):
    snapshot = json.loads((directory / "snapshot.json").read_text())
    ledger = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()]
    receipts = json.loads((directory / "photon_receipts.json").read_text())
    return {"turn_starts": sum(e["kind"] == "worker_started" for e in ledger),
            "deliveries": sum(bool(e["delivered_at"]) for e in snapshot["deliveries"]),
            "notifications": sum(r["status"] == "delivered" for r in receipts),
            "publications": len(snapshot["events"])}


async def execute_run(args):
    from .coordinator import Coordinator
    directory = Path(args.run_dir).resolve() if args.run_dir else ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    options = run_options(args, directory)
    notification = notifier_for(args, directory)
    controller = None
    try:
        controller = Coordinator(directory, mode=args.routing, endpoint=endpoint(args.endpoint), notifier=notification,
                                 context_budget=args.context_budget, **options)
        result = await controller.run()
    finally:
        if controller:
            controller.close()
        if notification:
            notification.close()
    from .evidence_audit import audit_run
    audit = audit_run(directory)
    atomic_json(directory / "audit.json", audit)
    print(json.dumps({"run_dir": str(directory), "core_result_passed": result["core_result_passed"],
                      "audit_passed": audit.get("passed"), "audit_path": str(directory / "audit.json"),
                      "concurrency": result.get("concurrency"),
                      "failed_checks": [k for k, v in audit.get("checks", {}).items() if v is False]}, indent=2))
    return 0 if audit.get("passed") else 1


async def execute_research(args):
    from .research import Research, DEFAULT_LIMITS, DEFAULT_MODELS, POLICY_VERSION, read_prompt_file
    config = None
    browser_options = bool(
        args.browser_cdp or args.browser_origin or args.browser_read_only
        or args.browser_unauth_cdp
        or args.max_browser_actions is not None or args.max_browser_replays is not None)
    if args.resume:
        if not args.run_dir:
            raise ValueError("--resume requires --run-dir")
        if (args.prompt or args.prompt_file or args.evidence or args.criterion
                or args.workers is not None or any((args.head_model, args.worker_model, args.qc_model))
                or browser_options):
            raise ValueError("Resume uses the saved prompt, models, workers, criteria, evidence, and browser policy; only limits may change")
    else:
        if args.prompt and args.prompt_file:
            raise ValueError("Supply either a prompt or --prompt-file, not both")
        prompt = read_prompt_file(args.prompt_file) if args.prompt_file else " ".join(args.prompt)
        if not prompt:
            if not sys.stdin.isatty():
                raise ValueError("Supply a quoted starting prompt or --prompt-file; interactive entry needs a terminal")
            prompt = input("Research objective: ")
        config = {"objective": prompt, "models": {
            "head": args.head_model or DEFAULT_MODELS["head"],
            "worker": args.worker_model or DEFAULT_MODELS["worker"],
            "qc": args.qc_model or DEFAULT_MODELS["qc"]},
            "workers": args.workers if args.workers is not None else 3,
            "criteria": args.criterion, "policy_version": POLICY_VERSION}
        if browser_options:
            endpoints, origins = args.browser_cdp or [], args.browser_origin or []
            if len(endpoints) != 2:
                raise ValueError("Browser research requires exactly two --browser-cdp endpoints")
            if not origins:
                raise ValueError("Browser research requires at least one exact --browser-origin")
            if config["workers"] < 2:
                raise ValueError("Browser research requires at least two workers for isolated sessions")
            if args.browser_read_only and args.max_browser_replays not in {None, 0}:
                raise ValueError("--browser-read-only cannot be combined with a positive replay budget")
            browser_config = {
                "endpoints": {"account_a": endpoints[0], "account_b": endpoints[1]},
                "worker_sessions": {"agent-a": "account_a", "agent-b": "account_b"},
                "allowed_origins": origins,
                "interaction_enabled": not args.browser_read_only,
                "request_replay_enabled": not args.browser_read_only,
                "max_actions_per_session": (args.max_browser_actions
                                            if args.max_browser_actions is not None else 24),
                "max_replays_per_session": (0 if args.browser_read_only
                                            else args.max_browser_replays
                                            if args.max_browser_replays is not None else 2),
                "script_eval_sessions": [],
            }
            if args.browser_unauth_cdp:
                if args.browser_read_only:
                    raise ValueError("--browser-unauth-cdp cannot be combined with --browser-read-only")
                if config["workers"] < 3:
                    raise ValueError("--browser-unauth-cdp requires three workers")
                browser_config["endpoints"]["unauth"] = args.browser_unauth_cdp
                browser_config["worker_sessions"]["agent-c"] = "unauth"
                browser_config["script_eval_sessions"] = ["unauth"]
            config["browser"] = browser_config
    directory = (Path(args.run_dir).expanduser().resolve() if args.run_dir else
                 ROOT / "runs" / ("research-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]))
    limits = {name: getattr(args, name) for name in DEFAULT_LIMITS if getattr(args, name) is not None}
    controller = Research(directory, config=config, limits=limits, evidence=args.evidence,
                          progress=lambda message: print(message, file=sys.stderr, flush=True))
    try:
        print(f"Research run: {directory}", file=sys.stderr, flush=True)
        result = controller.report() if args.dry_run else await controller.run()
        print(json.dumps({key: result[key] for key in ("run_dir", "status", "reason", "round", "phase", "models", "usage", "request_failures", "accepted_by_qc", "browser")}
                         | {"report": str(directory / "research-report.json")}, indent=2))
        return 0 if args.dry_run or result["status"] == "completed" else 2 if result["status"] == "paused" else 1
    finally:
        controller.close()


async def replay(args):
    from .coordinator import Coordinator
    directory = Path(args.run_dir).resolve()
    before = count_effects(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    args.photon = "real" if manifest.get("photon_enabled") else "off"
    notification = notifier_for(args, directory)
    controller = None
    try:
        controller = Coordinator(directory, mode=manifest["routing_mode"], endpoint=None, notifier=notification,
            worker_count=manifest.get("worker_count", 3), provider=manifest.get("provider", "codex"),
            model=manifest.get("requested_model", manifest.get("model")), context_budget=manifest.get("context_budget_per_worker"))
        if controller.store.run(controller.run_id)["status"] != "completed":
            raise ValueError("Replay is for completed runs; use run --run-dir to reconcile an interrupted run")
        await controller.run()
        after = count_effects(directory)
        record = {"before": before, "after": after, "no_duplicate_effects": before == after,
                  "process_restart": True, "mode": "fresh coordinator process reopens durable state; zero model starts"}
        atomic_json(directory / "replay.json", record)
    finally:
        if controller:
            controller.close()
        if notification:
            notification.close()
    from .evidence_audit import audit_run
    audit = audit_run(directory)
    atomic_json(directory / "audit.json", audit)
    print(json.dumps({"replay": record, "audit_passed": audit.get("passed"),
                      "audit_path": str(directory / "audit.json")}, indent=2))
    return 0 if record["no_duplicate_effects"] and audit.get("core_pass") else 1


def status(args):
    directory = Path(args.run_dir).resolve()
    if (directory / "research.json").exists():
        from .research import runtime_diagnostics
        state = json.loads((directory / "research.json").read_text())
        print(json.dumps({"run_dir": str(directory), **{key: state.get(key) for key in
                         ("run_id", "status", "reason", "round", "phase", "limits", "updated_at")},
                         "models": state["config"]["models"], "history": state["history"],
                         **runtime_diagnostics(directory),
                         "report": str(directory / "research-report.json")}, indent=2))
        return
    from .knowledge_store import KnowledgeStore
    state_file = directory / "fixture.json"
    if not state_file.exists():
        state_file = directory / "mission.json"
    fixture = json.loads(state_file.read_text())
    store = KnowledgeStore(directory / "data/knowledge.sqlite")
    try:
        run_id = fixture["run_id"]
        deliveries = store.deliveries(run_id)
        ledger = store.ledger(run_id)
        usage = {}
        for event in ledger:
            if event["kind"] == "token_usage":
                value = json.loads(event["data"])
                usage[value["agent"]] = value
        positions = {row["node_id"]: {"vector": json.loads(row["vector"]), "method": row["method"], "revision": row["version"]} for row in store.conn.execute("SELECT * FROM coordinates")}
        failures = [{"kind": row["kind"], "at": row["at"], "data": json.loads(row["data"])}
                    for row in ledger if row["kind"] in {"failure", "error", "tool_rejected"}]
        print(json.dumps({"run_id": run_id, "state": store.run(run_id)["status"], "workers": store.workers(run_id),
                          "positions": positions, "messages": [{k: r[k] for k in ("delivery_id", "event_id", "recipient", "state", "attempts", "accepted_at", "delivered_at", "acknowledged_at", "incorporated_at")} for r in deliveries],
                          "usage": usage, "failures": failures, "ledger_entries": len(ledger),
                          "report_path": str(directory / "report.json") if (directory / "report.json").exists() else None}, indent=2))
    finally:
        store.close()


def photon_test(args):
    from .photon_notifier import PhotonNotifier, NotificationPolicy
    run_id = args.run_id or str(uuid.uuid4())
    client = PhotonNotifier(ROOT / "data/notifications.sqlite", base_url=args.photon_url, state_path=args.photon_state,
                            policy=NotificationPolicy(quiet_start_hour=None, quiet_end_hour=None, min_interval_seconds=1))
    try:
        client.enqueue(run_id, "start", "Photon connection test for the local three-agent harness.", str(ROOT / "README.md"), key="photon-test:" + run_id)
        client.flush()
        receipts = client.receipts(run_id)
        print(json.dumps({"run_id": run_id, "receipts": receipts}, indent=2))
        return 0 if any(r["status"] == "delivered" for r in receipts) else 1
    finally:
        client.close()


def drain_notifications(args):
    from .knowledge_store import KnowledgeStore
    from .notification_pump import pump_once
    directory = Path(args.run_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if args.digest_interval is not None and args.digest_interval < 600:
        raise ValueError("Requested digest interval must be at least 600 seconds")
    args.photon = "real"
    notification = notifier_for(args, directory)
    store = KnowledgeStore(directory / "data/knowledge.sqlite")
    try:
        while True:
            result = pump_once(store, manifest["run_id"], notification, str(directory / "report.json"),
                               digest_interval=args.digest_interval)
            if args.once or result["receipt_count"]:
                print(json.dumps(result), flush=True)
            if args.once:
                return 0
            time.sleep(2)
    finally:
        store.close()
        notification.close()


def build_parser():
    parser = argparse.ArgumentParser(prog="hyperspace", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    workbench = commands.add_parser("workbench", help="Inspect harness architecture and saved runs in a local read-only workbench")
    workbench.add_argument("--port", type=int, default=8765, help="Loopback HTTP port (default: 8765)")
    chat = commands.add_parser("chat", help="Chat with your saved OpenRouter model in the Codex terminal")
    chat.add_argument("--provider", choices=("openrouter", "codex"), default="openrouter", help="Chat provider (default: openrouter)")
    chat.add_argument("prompt", nargs="?", help="Optional first message")
    chat.add_argument("--resume", nargs="?", const="last", help="Resume last, pick a session, or supply its ID")
    chat.add_argument("--model", help="Model ID; defaults to the saved API model for OpenRouter")
    chat.add_argument("--search", action="store_true", help="Enable native Codex web search")
    chat.add_argument("--no-alt-screen", action="store_true", help="Keep terminal scrollback")
    chat.add_argument("--add-dir", action="append", default=[], help="Additional directory the chat agent may edit")
    api = commands.add_parser("api", help="Privately configure the research workers' OpenRouter account")
    api_commands = api.add_subparsers(dest="api_command", required=True)
    setup = api_commands.add_parser("setup", help="Choose a model and enter your API key privately")
    setup.add_argument("--model", help="OpenRouter provider/model ID")
    api_commands.add_parser("status", help="Show configuration status without revealing the key")
    api_commands.add_parser("check", help="Check API authentication without generating model tokens")
    api_model = api_commands.add_parser("model", help="Change the research model, keeping the existing key")
    api_model.add_argument("model")
    research = commands.add_parser("research", help="Run GLM head, DeepSeek workers, and Qwen QC until accepted or bounded limits are reached")
    research.add_argument("prompt", nargs="*", help="Starting research prompt; prompts interactively when omitted")
    research.add_argument("--prompt-file", help="Read the starting prompt from a UTF-8 file")
    research.add_argument("--run-dir", help="Saved research directory; generated when omitted")
    research.add_argument("--resume", action="store_true", help="Resume a checkpoint; transient failures use fresh bounded phase attempts")
    research.add_argument("--dry-run", action="store_true", help="Save or inspect the configuration without model requests")
    research.add_argument("--evidence", action="append", default=[], metavar="FILE", help="Attach a UTF-8 evidence file (repeatable; contents are sent to OpenRouter)")
    research.add_argument("--criterion", action="append", default=[], help="An acceptance requirement Qwen must review (repeatable)")
    research.add_argument("--workers", type=int, choices=range(1, 4), help="DeepSeek workers (default: 3; plus GLM head and Qwen QC)")
    research.add_argument("--head-model", help="OpenRouter head model (default: deepseek/deepseek-v4-flash-0731)")
    research.add_argument("--worker-model", help="OpenRouter worker model (default: deepseek/deepseek-v4-flash)")
    research.add_argument("--qc-model", help="OpenRouter reviewer model (default: qwen/qwen3.8-flash)")
    research.add_argument("--max-rounds", type=int, help="Total round limit, including previous rounds (default: 8)")
    research.add_argument("--max-requests", type=int, help="Total provider request limit, including previous requests (default: 160)")
    research.add_argument("--max-minutes", type=int, help="Time limit for this invocation (default: 60)")
    research.add_argument("--max-phase-attempts", type=int, help="Fresh attempts allowed per phase, including prior attempts (default: 3)")
    research.add_argument("--request-timeout-seconds", type=int, help="Provider timeout for each request (default: 180)")
    research.add_argument("--browser-cdp", action="append", metavar="URL", help="Private CDP endpoint for an isolated owned test account (repeat exactly twice)")
    research.add_argument("--browser-unauth-cdp", metavar="URL", help="Private CDP endpoint for a dedicated unauthenticated session; enables model-authored JavaScript only there")
    research.add_argument("--browser-origin", action="append", metavar="ORIGIN", help="Exact authorized browser origin such as https://example.com (repeatable)")
    research.add_argument("--browser-read-only", action="store_true", help="Expose inspection and response reading without navigation, typing, clicking, or request replay")
    research.add_argument("--max-browser-actions", type=int, help="Durable browser actions per isolated session (default: 24; maximum: 100)")
    research.add_argument("--max-browser-replays", type=int, help="Observed same-origin GET/HEAD XHR replays per session (default: 2; maximum: 10)")
    run = commands.add_parser("run", help="Run the cooperation fixture (five workers by default)")
    run.add_argument("--routing", choices=["broadcast", "flat", "graph", "hyperbolic", "hybrid"], default="hybrid")
    run.add_argument("--run-dir")
    run.add_argument("--endpoint", default=os.environ.get("ASTRA_HYPERSPACE_ENDPOINT", "auto"))
    run.add_argument("--photon", choices=["real", "off"], default="off")
    run.add_argument("--context-budget", type=int, help="Peer-context tokens per worker; scales with count by default")
    rep = commands.add_parser("replay", help="Reopen a completed run and independently check duplicate effects")
    rep.add_argument("--run-dir", required=True)
    for name in ("status", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True)
    bench = commands.add_parser("benchmark", help="Execute the same live fixture in all five routing modes")
    bench.add_argument("--output-dir", default=str(ROOT / "runs" / ("benchmark-" + time.strftime("%Y%m%d-%H%M%S"))))
    bench.add_argument("--endpoint", default=os.environ.get("ASTRA_HYPERSPACE_ENDPOINT", "auto"))
    bench.add_argument("--reuse-hybrid")
    bench.add_argument("--previous-attempt", action="append", default=[])
    bench.add_argument("--retry-failed", action="store_true", help="Run one new bounded trial for a failed mode, preserving prior attempts")
    notify = commands.add_parser("photon-test")
    notify.add_argument("--run-id", help="Reuse to test notification deduplication")
    drain = commands.add_parser("notifications", help="Drain durable Photon backlog; optionally send requested count-only digests")
    drain.add_argument("--run-dir", required=True)
    drain.add_argument("--once", action="store_true")
    drain.add_argument("--digest-interval", type=int, help="Enable digests, interval in seconds (minimum 600)")
    phone = commands.add_parser("phone", help="Talk to the harness on Photon using the saved OpenRouter model")
    phone.add_argument("--directory", default=str(ROOT / "data" / "phone"))
    phone.add_argument("--endpoint", default=os.environ.get("ASTRA_HYPERSPACE_ENDPOINT", "auto"))
    phone.add_argument("--line-lock", default=os.path.expanduser("~/.config/hyperspace-harness/dispatch.lock"))
    phone_status = commands.add_parser("phone-status", help="Show private phone bridge activity metadata")
    phone_status.add_argument("--directory", default=str(ROOT / "data" / "phone"))
    mission = commands.add_parser("mission", help="Run 2..5 configured knowledge-worker tasks")
    mission.add_argument("config")
    mission.add_argument("--run-dir", required=True)
    mission.add_argument("--endpoint", default=os.environ.get("ASTRA_HYPERSPACE_ENDPOINT", "auto"))
    mission.add_argument("--photon", choices=["real", "off"], default="off")
    for command in (run, bench):
        command.add_argument("--workers", type=int, choices=range(2, 6), help="Worker count (default: five for new runs; maximum five)")
    for command in (run, bench, mission):
        command.add_argument("--provider", choices=["codex", "openrouter"], help="Model runtime (default: codex)")
        command.add_argument("--model", help="Exact model ID; OpenRouter also reads OPENROUTER_MODEL")
    for command in (run, rep, notify, mission, drain, phone):
        command.add_argument("--photon-url", default=os.environ.get("PHOTON_SIDECAR_URL", "http://127.0.0.1:8790"))
        command.add_argument("--photon-state", default=os.environ.get("PHOTON_RECIPIENT_STATE_PATH", os.path.expanduser("~/.config/hyperspace-harness/dispatch_state.json")))
    return parser


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(arguments or ["chat"])
    try:
        if args.command == "workbench":
            from .workbench import serve
            serve(ROOT, args.port)
            code = 0
        elif args.command == "chat":
            from .chat_cli import launch_chat
            code = launch_chat(args, ROOT)
        elif args.command == "api":
            from .api_settings import api_setup, api_status, api_set_model, api_check
            if args.api_command == "setup":
                result = api_setup(args.model)
            elif args.api_command == "model":
                result = api_set_model(args.model)
            elif args.api_command == "check":
                result = api_check()
            else:
                result = api_status()
            print(json.dumps(result, indent=2))
            code = 0 if args.api_command != "check" or result.get("key_valid") is True else 1
        elif args.command == "run":
            code = asyncio.run(execute_run(args))
        elif args.command == "research":
            code = asyncio.run(execute_research(args))
        elif args.command == "replay":
            code = asyncio.run(replay(args))
        elif args.command == "status":
            status(args)
            code = 0
        elif args.command == "audit":
            from .evidence_audit import audit_run
            result = audit_run(args.run_dir)
            print(json.dumps(result, indent=2))
            code = 0 if result.get("passed") else 1
        elif args.command == "benchmark":
            from .benchmark import run_benchmark
            options = run_options(args)
            result = asyncio.run(run_benchmark(args.output_dir, endpoint(args.endpoint), hybrid_run=args.reuse_hybrid,
                                              previous_attempts=args.previous_attempt, retry_failed=args.retry_failed, **options))
            print(json.dumps({"status": result.get("status"), "all_core_pass": result.get("all_core_pass", False),
                              "report": str(Path(args.output_dir).resolve() / "benchmark.json"),
                              "modes": {mode: row.get("audit", {}).get("core_pass") for mode, row in result.get("runs", {}).items()},
                              "execution_error": result.get("execution_error")}, indent=2))
            code = 0 if result.get("all_core_pass", False) else 1
        elif args.command == "mission":
            from .mission import run_mission
            from .runtime_factory import runtime_options
            saved_path = Path(args.run_dir) / "mission.json"
            saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
            provider, model = runtime_options(args.provider or saved.get("provider", "codex"), args.model or saved.get("model"))
            notification = notifier_for(args, Path(args.run_dir))
            try:
                result = asyncio.run(run_mission(args.config, args.run_dir, endpoint(args.endpoint), notification,
                                                provider=provider, model=model))
            finally:
                if notification:
                    notification.close()
            print(json.dumps({"run_id": result["run_id"], "protocol_passed": result.get("protocol_passed", False),
                              "semantic_correctness_verified": result.get("semantic_correctness_verified", False),
                              "concurrency": result.get("concurrency"),
                              "report": str(Path(args.run_dir).resolve() / "report.json")}, indent=2))
            code = 0 if result.get("protocol_passed", False) else 1
        elif args.command == "notifications":
            code = drain_notifications(args)
        elif args.command == "phone":
            from .phone_bridge import serve
            asyncio.run(serve(args.directory, None, base_url=args.photon_url,
                              state_path=args.photon_state, line_lock=args.line_lock))
            code = 0
        elif args.command == "phone-status":
            print((Path(args.directory) / "status.json").read_text())
            code = 0
        else:
            code = photon_test(args)
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        # Credentials never enter these configuration errors. Provider response
        # bodies are intentionally not printed by the notifier/backend.
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        code = 1
    raise SystemExit(code)
