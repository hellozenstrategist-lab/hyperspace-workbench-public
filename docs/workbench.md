# Hyperspace Workbench

The Workbench is a local browser view of the harness architecture, configuration
defaults, and saved mission and research evidence. Browsing does not start workers,
contact providers, or change saved runs. The optional assistant defaults to
**ChatGPT Sub** and generates a response only when you send a message. Explicit
model-list and connection-status checks may contact a provider without generation.

From the source tree:

```bash
./harness workbench --port 8765
```

Open `http://127.0.0.1:8765` in a browser. Stop the server with Ctrl+C. The standalone
entrypoint is `python -m astra_harness.workbench --port 8765`. The server binds only
to `127.0.0.1`; there is no public hosting or remote access setting.

## Workbench controls

- **Assembly:** switch between mission and research architecture. Drag components
  on the grid, pan the canvas, and scroll to zoom. Spread out separates the parts;
  Focus part keeps the selected part and its immediate connections. Double-click a
  part to isolate it. Use `F` to fit, `E` to explode, `I` to isolate, and Escape to
  return. Arrow keys move the selected part; Shift uses larger steps. Shift-click
  selects up to eight parts for one conversation.
- **Box actions:** click a box for the illustrated action menu: Connect,
  Disconnect, Call AI, Delete, or Duplicate. Connection actions let you choose
  endpoints. Every graph edit opens a preview before applying. Delete hides
  source-backed parts only in the visual draft; it never deletes their files.
  Press Enter on a focused box to open the menu, use arrow keys to move between
  actions, and Escape to close it.
- **Call AI:** opens Ask, Inspect, Refactor, Run preview, Explain, and Generate
  worker. Inspect reads source; the other actions open chat with an editable
  prompt. **Run preview** and **Generate worker** only prefill prompts for a
  hypothetical walkthrough or visual worker design. They never start the harness;
  you still choose whether to send the prompt.
- **Inspector:** select a component to see its responsibilities, inputs, outputs,
  connected parts, and design constraints. Source displays the actual allowlisted
  module and can expand into a larger reader. Data shows the selected recorded
  event and matching events. The close button hides the inspector; selecting a
  part opens it again. The AI tab edits a visual-only part profile. `/` focuses
  component search.
- **Layouts:** positions, draft connections, and component notes persist in this
  browser. Save workspace exports JSON; Open workspace restores it. Undo and redo
  include imported positions, notes, draft graphs, part AI profiles, and the active
  scene. These are workspace edits, not changes to the executable harness or run
  configuration.
- **Dashboard:** filter saved runs, inspect worker and research-role states,
  compare recorded counts, inspect peer receipts, and read saved artifacts.
  Research lists head and QC separately from its parallel workers.
- **Geometry:** inspect actual saved mission coordinates and select points or
  their nearest neighbors. Distances are calculated locally using the exact
  Poincaré reference formula. Research runs display their separate artifact-based
  storage boundary instead of inventing geometry.
- **Recorded events:** choose a run and scrub or play its saved events. Playback
  animates the mapped architecture; it does not execute the workflow. Refresh data
  polls saved files at the configured interval while the tab is visible and playback
  and field editing are inactive. Manual refresh is always available.
- **Settings:** choose the assistant provider and model, manage attached context,
  and adjust browser-local workspace preferences.

## AI settings

**ChatGPT Sub** is the first/default provider. It uses the installed Codex CLI's
native ChatGPT login; credentials are not copied into the project. Leave its model
blank for the account default, or enter an exact model ID. If disconnected, run
`codex login` in a terminal and choose ChatGPT. No OpenAI API key is required.

**OpenRouter** requires an explicit model and its own API key. Requests have
separate OpenRouter billing and are not covered by the ChatGPT subscription.
Set its maximum output to 256–8192 tokens, defaulting to 2048. A supplied key is
session-only by default. **Remember on this machine** optionally writes it to a
plaintext local credential file with owner-only `0600` permissions; it is not
encrypted. Existing environment credentials are recognized. Clear removes the
configured session/saved key, but an environment-provided key can remain available.
Password fields clear after submission, and credentials never enter browser storage.

**Save AI settings** selects the actual assistant route. There is no automatic
provider/model fallback. Route changes are blocked while an assistant reply is
pending. **Load models** and **Check active connection** are manual checks without
generation; opening or refreshing saved-data views does not make these requests.

Context preferences independently enable component-source excerpts and the selected
run's context. Review the resulting attachment tray before sending. Workspace
preferences apply immediately in this browser: grid snapping, the quick guide,
decorative artwork, reduced motion, and automatic saved-data refresh at 5, 15, 30,
or 60 seconds. Automatic refresh defaults off; its default interval is 15 seconds.
Reset workspace preferences does not change AI credentials or provider settings.

## AI assistant

Right-click a part, choose **Ask AI**, or use **Call AI** in its action menu.
The chat shows attached parts, the matching recorded run/event, and any highlighted
text. Expand **Included context** before sending. Source and run attachments follow
the privacy choices in Settings. The server adds bounded, redacted catalog context
and enabled source excerpts and immediate connections; workspace draft
positions, notes, and recent conversation history are also included. Changing
selection updates the context for your next message, not an in-flight request.

The panel displays the active provider and resolved model. Opening chat checks the
configured connection without generation. The workbench's provider settings are
separate from the mission runtime and research-role model configuration.

Ask for explanations or visual workflow changes. A proposal lists each change;
**Preview changes** displays a temporary candidate without saving it. Choose
**Apply draft** to save it locally, **Dismiss preview** to discard it, or **Undo**
after applying. **Reset draft** returns the scene to its source-backed topology.
Draft stages are visibly labeled and have no executable implementation. Neither
manual actions nor the assistant can modify harness source, run configuration,
saved evidence, or start actual workflows. The assistant has no shell, browser,
connector, or source-editing tools.

The inspector's **AI** tab records a part's enabled flag, primary/fallback model
names, maximum steps, and temperature as **visual design settings**. Preview them,
apply the draft, and use Undo to restore the previous profile. These profiles are
included in workspace exports; they do not select the actual chat model, activate
fallback routing, or execute steps. Use global Settings to change the real assistant
connection.

**Stop** cancels a pending generation. Closing the panel preserves the conversation
and lets a pending response finish. The browser retains this tab's recent chat;
the backend stores bounded job/conversation history in `data/workbench-chat/`.
Requests are recorded before generation and are not automatically replayed after
an uncertain interruption or server restart. If the browser cannot confirm a
submission, **Recover request** checks using its exact original message and ID,
preventing duplicate generation for an already accepted request. New messages
wait until the pending request is resolved. Only one generation runs at a time.
Chat history can contain private project information even after secret redaction.

## Saved data

The reader discovers saved run directories beneath `runs/`, `research-runs/`, and
`data/research/`, scanning at most four directory levels. Mission views combine
saved JSON metadata with the SQLite knowledge graph and delivery ledger. Research
views show saved configuration, phase attempts, model requests, reported usage,
and evidence artifacts. A refresh reads the current saved files. It does not
connect to HyperspaceDB, Photon, CDP sessions, or a model provider.

SQLite inspection copies bounded database and WAL bytes into a temporary
directory and opens that snapshot using a read-only SQLite URI and `query_only`.
This avoids creating or updating sidecar files beside the original database.
Concurrent writes can produce an incomplete snapshot; the UI reports a warning
and a later refresh can read it again. Malformed or partially written JSON also
produces a warning instead of stopping the run listing.

The architecture catalog allowlists source files. Artifacts are selected from
known run metadata, phase results, and evidence directories. The server does not
provide arbitrary filesystem browsing and rejects symbolic links. Source and
artifact previews are limited to 128 KiB; large previews report truncation.
Graph, ledger, delivery, and artifact views show at most 500 records. Database
counts and saved usage metrics describe their respective recorded totals, even
when the visible record list is capped. Unknown or incomplete usage remains
identified in the saved metrics.

## Interpretation and access

Mission receipts establish delivery, acknowledgement, and declared use. They do
not establish semantic correctness. Research QC acceptance is a recorded model
judgment. Worker lifetime and HTTP request overlap are separate measurements;
neither establishes provider GPU generation concurrency.

The server accepts loopback Host headers and rejects cross-origin requests. It
sends no CORS permissions, disables caching and framing, and serves only
allowlisted application assets. Assistant POSTs additionally require a same-origin
Origin header and a current server-issued request nonce. API responses and previews redact credential fields,
authorization and cookie headers, and recognizable secret token forms. Saved
free text can still contain private project information: use the Workbench only
on a trusted local machine and avoid sharing previews indiscriminately.

Saved harness inspection remains GET-only:

- `/api/overview` returns the catalog, defaults, diagnostics, and run summaries.
- `/api/runs` returns saved run summaries and warnings.
- `/api/runs/<id>` returns details for an opaque run ID.
- `/api/runs/<id>/artifacts/<artifact-id>` returns a bounded artifact preview.
- `/api/components/<id>/source` returns a catalogued source preview.

Assistant endpoints are separate:

- `GET /api/assistant/status` reports the selected provider's connection and request nonce.
- `GET /api/assistant/settings` returns public provider, model, credential-presence,
  and context settings, never a credential value.
- `POST /api/assistant/settings` saves validated AI settings and optional credential
  changes using the same origin/nonce protection as other assistant POSTs.
- `GET /api/assistant/models/<provider>` explicitly loads model choices for
  `chatgpt` or `openrouter` without generating a reply.
- `POST /api/assistant/messages` validates context and creates an asynchronous job.
- `GET /api/assistant/jobs/<id>` returns job status and a validated visual proposal.
- `POST /api/assistant/jobs/<id>/cancel` cancels a pending response.
- `GET /api/assistant/history` returns bounded stored conversation history.

There is no server-side apply/source-write endpoint. The browser's preview/apply
operations only update its visual workspace.

The Python interface is `WorkbenchReader(root)` with `overview()`, `list_runs()`,
`run_detail(id)`, `artifact(id, artifact_id)`, and `source(component_id)` methods.
`create_server(root, port=8765)` constructs the server without starting its loop.

## Verification

Reader/API tests use temporary fixtures. Assistant and OpenRouter tests inject fake
transports; browser assistant/settings tests intercept their endpoints and do not
generate model traffic. The focused Python regression suite contains 200 tests.
No live provider-generation result is bundled or claimed by this repository.

Run the Python reader, HTTP, and catalog tests with:

```bash
.venv/bin/python -m pytest tests/test_workbench.py tests/test_workbench_chat.py tests/test_workbench_settings.py
```

`tests/workbench_browser.mjs` validates the empty-inventory UI in a clean checkout.
When saved synthetic fixtures are present, it additionally checks dragging,
undo/redo, isolation, source expansion, workspace import/export, notes, recorded
playback, dashboard filters and comparisons, artifact previews, geometry, and a
1024-pixel viewport. It makes no model calls. It requires Node, Playwright, and
installed Google Chrome; Playwright can be installed outside the harness:

```bash
npm install --prefix /tmp/hyperspace-workbench-qa playwright
WORKBENCH_PLAYWRIGHT_MODULE=/tmp/hyperspace-workbench-qa/node_modules/playwright/index.mjs \
  node tests/workbench_browser.mjs
WORKBENCH_PLAYWRIGHT_MODULE=/tmp/hyperspace-workbench-qa/node_modules/playwright/index.mjs \
  node tests/workbench_assistant_browser.mjs
WORKBENCH_PLAYWRIGHT_MODULE=/tmp/hyperspace-workbench-qa/node_modules/playwright/index.mjs \
  node tests/workbench_settings_browser.mjs
```

`WORKBENCH_URL` overrides the loopback URL. `WORKBENCH_BROWSER_ARTIFACTS` overrides
the screenshot and JSON report directory, defaulting to
`/tmp/hyperspace-workbench-qa`. Browser checks require a running workbench and
its saved acceptance fixture; the Python tests create their own temporary data.

The assistant browser suite checks right-click/multi-part context, inert reply
rendering, preview/dismiss/apply/undo/redo, reload persistence, every box-menu
action, keyboard access, cancellation, and compact layouts.
The settings suite checks provider/model selection, credential handling, pending-reply
locks, local preference persistence, Call AI actions, and visual part-profile
preview/apply/undo without dispatching a worker or changing source.
