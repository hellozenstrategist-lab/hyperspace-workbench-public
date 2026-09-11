"""Non-secret source provenance for a source-tree harness invocation."""
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "astra_harness").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    files += [root / name for name in ("harness", "requirements.lock.txt", "pyproject.toml")]
    hashes = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in files}
    versions = {}
    for name in ("grpcio", "protobuf", "numpy", "cryptography", "pytest", "hypothesis"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return {"schema_version": 1, "source_sha256": hashes, "installed_dependencies": versions,
            "scope": "Source files on disk at invocation; no environment values or authentication files."}
