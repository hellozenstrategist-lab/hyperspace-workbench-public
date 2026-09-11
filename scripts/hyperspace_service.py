#!/usr/bin/env python3
"""Start/status/stop the bounded persistent local HyperspaceDB service.

No public ports, host mounts, inherited credentials, or external service calls.
Requires Linux Docker bridge access from the host. Stop preserves the volume.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request

CONTAINER = "astra-harness-db"
NETWORK = "astra-harness-net"
VOLUME = "astra-harness-data"
LABEL = "org.astra-harness.local"
IMAGE = "glukhota/hyperspace-db@sha256:e3000c0afb1e13a880b5250499f36876e2af385b733a4fa62b10dff5bbbe2a2e"


def docker(*args, missing_ok=False):
    # An empty scoped Docker config prevents inherited registry credentials.
    with tempfile.TemporaryDirectory(prefix="astra-docker-") as config:
        result = subprocess.run(["docker", "--config", config, "--host", "unix:///var/run/docker.sock", *args], capture_output=True, text=True, env={"PATH": os.defpath}, timeout=55)
    if result.returncode:
        if missing_ok and ("no such" in result.stderr.lower() or "not found" in result.stderr.lower()):
            return None
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def inspect(kind, name):
    raw = docker(kind, "inspect", name, missing_ok=True)
    return json.loads(raw)[0] if raw else None


def owned(info, kind):
    labels = info.get("Config", {}).get("Labels", {}) if kind == "container" else info.get("Labels", {})
    if not labels or labels.get(LABEL) != "true":
        raise RuntimeError(f"existing {kind} has no harness ownership label; leaving it untouched")


def status():
    info = inspect("container", CONTAINER)
    if info is None:
        return {"container": CONTAINER, "exists": False, "running": False, "volume": VOLUME}
    owned(info, "container")
    ip = info["NetworkSettings"]["Networks"].get(NETWORK, {}).get("IPAddress")
    return {"container": CONTAINER, "exists": True, "running": info["State"]["Running"], "status": info["State"]["Status"], "exit_code": info["State"]["ExitCode"], "volume": VOLUME, "network": NETWORK, "image": info["Config"]["Image"], "endpoint": f"{ip}:50051" if ip else None, "http_endpoint": f"http://{ip}:50050" if ip else None, "published_ports": info["HostConfig"].get("PortBindings") or {}, "stop_signal": info["Config"].get("StopSignal")}


def start():
    existing = inspect("container", CONTAINER)
    if existing:
        owned(existing, "container")
        if existing["Config"]["Image"] != IMAGE:
            raise RuntimeError("existing harness container uses a different image; migration must be explicit")
        if not existing["State"]["Running"]:
            docker("start", CONTAINER)
    else:
        network = inspect("network", NETWORK)
        if network:
            owned(network, "network")
            if not network.get("Internal"):
                raise RuntimeError("harness network must be internal")
        else:
            docker("network", "create", "--internal", "--label", LABEL + "=true", NETWORK)
        volume = inspect("volume", VOLUME)
        if volume:
            owned(volume, "volume")
        else:
            docker("volume", "create", "--label", LABEL + "=true", VOLUME)
        args = ["run", "-d", "--name", CONTAINER, "--label", LABEL + "=true", "--network", NETWORK, "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", "256", "--memory", "2g", "--cpus", "2", "--stop-signal", "SIGINT", "--stop-timeout", "30", "--mount", "type=volume,src=" + VOLUME + ",dst=/app/data", "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m"]
        settings = {"HS_DATA_DIR": "/app/data", "HS_DIMENSION": "2", "HS_METRIC": "poincare", "HS_QUANTIZATION_LEVEL": "none", "HS_MAX_RAM_GB": "1", "HS_WAL_SEGMENT_SIZE_MB": "16", "HYPERSPACE_WAL_SYNC_MODE": "strict", "HYPERSPACE_SNAPSHOT_INTERVAL_SEC": "60", "HS_FAST_UPSERT_DELTA": "false", "HYPERSPACE_EMBED": "false", "HS_GOSSIP_ENABLED": "false", "HS_GOSSIP_PEERS": "", "HS_REPLICATION_ALLOWED": "false", "RUST_LOG": "warn"}
        for key, value in settings.items():
            args.extend(["-e", key + "=" + value])
        docker(*args, IMAGE)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 30
    last_error = None
    while time.monotonic() < deadline:
        result = status()
        if not result["running"]:
            raise RuntimeError("server stopped during startup: " + json.dumps(result))
        try:
            with opener.open(result["http_endpoint"] + "/api/health", timeout=1) as response:
                result["http_health"] = json.loads(response.read())
                return result
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.2)
    raise RuntimeError("server health did not become available: " + str(last_error))


def stop():
    info = inspect("container", CONTAINER)
    if info:
        owned(info, "container")
        if info["State"]["Running"]:
            docker("stop", "--time", "30", CONTAINER)
    result = status()
    result["persistent_volume_preserved"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop"))
    arguments = parser.parse_args()
    print(json.dumps(globals()[arguments.command](), indent=2))
