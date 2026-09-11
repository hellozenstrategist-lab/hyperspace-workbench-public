# Interactive harness chat

Run `hyperspace` or `hyperspace chat` in a terminal. From the source directory, `./harness chat` does the same thing. This opens the native Codex interface using your saved OpenRouter model and API key. The assistant can edit files, run commands, change mission configuration, and operate the harness within Codex's workspace sandbox and approval settings. Research runs remain limited to five workers. Chat requests consume your OpenRouter allowance.

```bash
hyperspace chat "Inspect the harness and help me plan a research mission"
hyperspace chat --resume
hyperspace chat --resume pick
hyperspace chat --resume SESSION_ID "Continue the previous task"
hyperspace chat --provider codex --model example/alternate-model --search --no-alt-screen
hyperspace chat --add-dir /path/to/another/project
```

Without `--model`, chat uses the saved OpenRouter model. Use `--provider codex` for your ChatGPT login and native model preference. `--resume` selects the most recent session for this workspace; `--resume pick` opens the native session picker. With either option, enter your follow-up after the session opens. To supply an initial follow-up on the command line, pass an explicit session ID. `--add-dir` grants write access to that existing directory. The launcher passes arguments directly to Codex without a shell.

The wrapper sets `workspace-write` and approvals `on-request`. OpenRouter uses a custom Responses API provider. The saved key is loaded internally into the child process environment, never command arguments, and excluded from shell tool environments. Native web search (`--search`) requires `--provider codex`. Only that explicit provider sets `forced_login_method="chatgpt"`; run `codex login` if it needs authentication. See [the configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

Bounded harness guidance is supplied through the documented `developer_instructions` configuration setting for this invocation. It replaces an existing value of that setting; ordinary workspace instructions, skills, and native tools remain available. It describes the five-worker limit, relevant source files, and safe configuration commands. See [the configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

On first launch, native Codex can ask whether you trust this project directory.
Use a normal interactive terminal. The launcher was checked through the
project-trust screen without submitting a model task.

## OpenRouter configuration

The interactive chat defaults to OpenRouter. Research workers separately use OpenRouter when a run explicitly selects `--provider openrouter`.

```bash
hyperspace api setup --model provider/model
hyperspace api status
hyperspace api model provider/another-model
hyperspace api check
```

Replace `provider/model` with the exact OpenRouter model ID you intend to use. Bare `hyperspace api setup` also prompts for the model when none is configured. Initial setup uses a hidden terminal prompt when no key is already available. Enter the key there, never in a chat or command argument. Setup can also save an existing `OPENROUTER_API_KEY` environment value. Noninteractive setup with an environment key and no model saves the key only; configure the model separately before running research. The assistant can later change the model or check configuration without displaying the credential.

Settings are saved outside the project at `~/.config/hyperspace-harness/openrouter.json`; its directory uses mode `700` and its file mode `600`. Symlinked or insecure settings are rejected. `api status`, `api setup`, and `api model` return only the effective model, credential presence, and sources. Environment values `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` override saved values independently. Changing the saved model preserves the saved key. `HYPERSPACE_HARNESS_CONFIG` overrides the settings file location for isolated environments and tests.

`api check` makes a bounded account-key request to OpenRouter, reports validity/connectivity, and never starts a generation. It does not establish model access, tool support, quota for a particular task, or research quality. Research runs use the selected provider's account allowance; choose tasks and model explicitly before starting them. Offline tests replace both native process launch and HTTP requests, so they do not verify a live login or provider response.
