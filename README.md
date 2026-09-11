# Hyperspace Workbench

A local-first visual workbench for exploring software harnesses as an interactive system diagram. Select components, inspect bounded source, review recorded data, and prepare AI-assisted visual drafts with preview and undo.

This repository is intentionally distributed with **no runs, databases, chat history, credentials, artifacts, or browser state**. On first launch the workbench shows an empty saved-data inventory and creates local state only when you use it.

## Run locally

Requirements: Python 3.11 or newer. The workbench uses the standard library and does not require a model key to browse the UI.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
./harness workbench --port 8765
```

Open <http://127.0.0.1:8765>. If your Python executable has another name, set `PYTHON_BIN` when using `./harness`.

The server binds to loopback only. It reads source files from this checkout and optional run directories beneath `runs/`, `research-runs/`, and `data/research/`; those directories are absent in this distribution, so the initial dashboard is clean. Add your own saved run exports locally if you want the dashboard to display them.

## AI assistant

ChatGPT Sub is the default provider. It uses the installed Codex login and does not require an API key. OpenRouter is supported as an explicit alternative from **Settings**; it requires your own model ID and API key, with separate provider billing. Keys are session-only by default. If you choose to remember one, it is written locally with owner-only permissions and is never committed or returned to the browser.

Right-click a component or use **Call AI** to open Ask, Inspect, Refactor, Run preview, Explain, and Generate worker actions. Run preview and Generate worker only prepare prompts. Assistant proposals are visual drafts: review, apply, dismiss, or undo them; they never edit source files or execute a worker. The inspector’s AI tab stores per-part model settings as visual design metadata, not active routing.

## Privacy and local state

Settings let you control whether source excerpts and selected run context are attached to assistant requests. Workspace preferences (snap-to-grid, guides, artwork, motion, and refresh) stay in browser storage. Server-side assistant settings and bounded conversation history are created under `data/workbench-chat/` only after use. Keep that directory private and do not commit it.

The `.gitignore` excludes runtime state, credentials, databases, caches, and generated reports. Review `git status` before publishing a fork.

## Development

Install test dependencies and run the unit suite:

```bash
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

The browser suites require Node, Playwright, and Chrome. They use temporary fixtures and intercepted assistant endpoints; they do not make model calls. See [docs/workbench.md](docs/workbench.md) for the interaction model, API boundaries, settings details, and browser commands.

## Safety boundary

The assistant is intentionally tool-free. It can explain attached context and propose reversible diagram changes, but it cannot browse, execute shell commands, edit source, start workers, or claim fresh run results. Provider credentials stay server-side.

## License

No license is granted for this repository. All rights are reserved unless a license is added later.
