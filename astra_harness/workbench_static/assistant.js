(() => {
  'use strict';

  const select = selector => document.querySelector(selector);
  const panel = select('#assistant-panel');
  const bridge = () => window.hyperspaceWorkbench;
  const storageKey = 'hyperspace-assistant-v1';
  const state = {context: null, conversationId: null, job: null, pending: null, submitting: false, messages: [], csrf: null, available: false, busy: false, polling: false, returnFocus: null};
  state.provider = 'chatgpt';
  state.label = 'ChatGPT Sub';
  state.privacy = {include_source: true, include_run: true};
  state.routing = JSON.stringify(['chatgpt', '', true, true]);
  const recover = element('button', 'secondary-button', 'Recover request');
  recover.id = 'assistant-recover';
  recover.type = 'button';
  recover.hidden = true;
  select('#assistant-stop').before(recover);

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function record(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
  function identifier(value) { return typeof value === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(value); }
  function textWithin(value, maximum) { return typeof value === 'string' && value.length <= maximum; }
  function sceneId(value) { return value === 'mission' || value === 'research'; }

  function validProposal(value) {
    if (!record(value) || !textWithin(value.title, 160) || !textWithin(value.description, 1600) || !Array.isArray(value.operations) || !value.operations.length || value.operations.length > 12 || JSON.stringify(value).length > 16000) return false;
    const fields = {move_component: ['component_id', 'x', 'y'], set_note: ['component_id', 'text'], connect: ['source', 'target', 'label', 'kind'], disconnect: ['source', 'target'], add_component: ['component_id', 'name', 'description', 'category', 'x', 'y'], remove_component: ['component_id']};
    return value.operations.every(operation => {
      if (!record(operation) || !Object.hasOwn(fields, operation.type)) return false;
      const required = fields[operation.type];
      if (Object.keys(operation).length !== required.length + 1 || required.some(key => !Object.hasOwn(operation, key))) return false;
      if (['component_id', 'source', 'target'].some(key => Object.hasOwn(operation, key) && !identifier(operation[key]))) return false;
      if (Object.hasOwn(operation, 'x') && (!Number.isFinite(operation.x) || !Number.isFinite(operation.y) || operation.x < 0 || operation.x > 2400 || operation.y < 0 || operation.y > 1600)) return false;
      if (operation.type === 'set_note' && !textWithin(operation.text, 4000)) return false;
      if (operation.type === 'connect' && (!textWithin(operation.label, 160) || !['control', 'data', 'feedback'].includes(operation.kind))) return false;
      if (operation.type === 'add_component' && (!/^draft-[a-z0-9][a-z0-9-]{0,53}$/.test(operation.component_id) || !textWithin(operation.name, 80) || !operation.name.trim() || !textWithin(operation.description, 1600) || !['control', 'runtime', 'storage', 'routing', 'interface', 'research'].includes(operation.category))) return false;
      return operation.type !== 'remove_component' || operation.component_id.startsWith('draft-');
    });
  }

  function savedMessage(value) {
    if (!record(value) || !['user', 'assistant', 'system'].includes(value.role) || !textWithin(value.text, 8000)) return null;
    const message = {role: value.role, text: value.text};
    if (textWithin(value.label, 80)) message.label = value.label;
    if (textWithin(value.context, 4000)) message.context = value.context;
    if (value.role === 'assistant' && sceneId(value.scene) && validProposal(value.proposal)) {
      message.proposal = value.proposal;
      message.scene = value.scene;
      message.dismissed = value.dismissed === true;
    }
    return message;
  }

  function validPending(value) {
    if (!record(value) || !record(value.body) || !sceneId(value.scene)) return false;
    const body = value.body;
    if (Object.keys(body).length !== 5 || !identifier(body.request_id) || (body.conversation_id !== null && !identifier(body.conversation_id)) || !textWithin(body.message, 6000) || !body.message.trim() || !record(body.selection) || !record(body.workspace)) return false;
    const selection = body.selection;
    return selection.scene === value.scene && Array.isArray(selection.component_ids) && selection.component_ids.length <= 8 && selection.component_ids.every(identifier) && (selection.run_id === null || identifier(selection.run_id)) && (selection.event_id === null || textWithin(selection.event_id, 500)) && textWithin(selection.selected_text, 4000) && new TextEncoder().encode(JSON.stringify(body)).length <= 128 * 1024;
  }

  function persist() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify({conversationId: state.conversationId, job: state.job, pending: state.pending, messages: state.messages.slice(-30), routing: state.routing}));
      return true;
    } catch { return false; }
  }

  function showError(message = '') {
    select('#assistant-error').textContent = message;
    select('#assistant-error').hidden = !message;
  }

  function setBusy(busy) {
    state.busy = busy;
    const previewing = Boolean(bridge()?.hasPreview?.());
    select('#assistant-send').disabled = busy || !state.available || previewing;
    select('#assistant-stop').hidden = !busy || !state.job;
    recover.hidden = !state.pending || state.submitting;
    recover.disabled = state.submitting;
    select('#assistant-new').disabled = busy;
    select('#assistant-input').setAttribute('aria-busy', String(busy));
    panel.classList.toggle('is-busy', busy);
    select('#assistant-status').textContent = state.pending && !state.submitting ? 'Request unconfirmed · recover manually' : busy ? 'Thinking · visual drafts only' : previewing ? 'Apply or dismiss preview before sending' : state.available ? state.provider === 'openrouter' ? 'Configured · OpenRouter' : 'Connected · ChatGPT subscription' : 'Not connected';
    window.dispatchEvent(new CustomEvent('workbench:assistant-state', {detail: {busy}}));
    const notice = select('#assistant-draft-status');
    if (previewing) {
      notice.dataset.preview = 'true';
      notice.hidden = false;
      notice.textContent = 'Apply draft or Dismiss preview before sending another message. Chat uses your applied visual workspace.';
    } else if (notice.dataset.preview === 'true') {
      delete notice.dataset.preview;
      notice.hidden = true;
    }
  }

  async function request(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? 'GET' : 'POST',
        headers: body === undefined ? {} : {'Content-Type': 'application/json', 'X-Workbench-CSRF': state.csrf},
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      let result;
      try { result = await response.json(); }
      catch {
        const error = new Error('The local assistant returned an unreadable response.');
        error.status = response.status;
        throw error;
      }
      if (!response.ok) {
        const error = new Error(typeof result?.error === 'string' ? result.error : 'The local assistant could not complete this request.');
        error.status = response.status;
        throw error;
      }
      return result;
    } finally {
      clearTimeout(timer);
    }
  }

  async function checkConnection() {
    state.csrf = null;
    try {
      const status = await request('/api/assistant/status');
      updateProvider(status);
      state.csrf = status.csrf;
      state.available = Boolean(status.available && status.authenticated);
      select('#assistant-model').textContent = status.model || 'ChatGPT account default';
      select('#assistant-model').title = status.cli_version || '';
      showError(state.available ? '' : status.error || (state.provider === 'openrouter' ? 'Choose an OpenRouter model and add its API key in Settings.' : 'Sign in on this machine with codex login, then reopen the assistant. No API key is needed.'));
    } catch (error) {
      state.available = false;
      showError(error.message);
    }
    setBusy(state.busy);
    if (state.pending && state.csrf) showError('This request has not been confirmed. Recover request sends its original body and ID; nothing is retried automatically.');
  }

  function updateProvider(status) {
    const provider = status.provider === 'openrouter' ? 'openrouter' : 'chatgpt';
    const privacy = {include_source: status.privacy?.include_source !== false, include_run: status.privacy?.include_run !== false};
    const routing = JSON.stringify([provider, status.configured_model || '', privacy.include_source, privacy.include_run]);
    if (!state.busy && state.routing !== routing) {
      const hadConversation = Boolean(state.conversationId || state.messages.length);
      state.conversationId = null;
      state.messages = [];
      state.routing = routing;
      if (hadConversation) appendMessage({role: 'system', text: 'AI connection or privacy changed. A new conversation starts here; previous messages are not forwarded.'});
      persist();
      renderMessages();
    }
    state.provider = provider;
    state.label = provider === 'openrouter' ? 'OpenRouter' : 'ChatGPT Sub';
    state.privacy = privacy;
    select('#assistant-title').textContent = state.label;
    panel.setAttribute('aria-label', state.label + ' assistant');
    select('#assistant-open').setAttribute('aria-label', 'Ask AI with ' + state.label);
    select('.assistant-provider').textContent = state.label;
    select('.assistant-privacy').textContent = provider === 'openrouter' ? 'Uses your OpenRouter API key. Separate provider charges apply. Only reviewed context is sent.' : 'Uses your ChatGPT subscription. Only reviewed context is sent.';
    attachContext(bridge()?.context(state.context?.selection.selected_text || ''));
  }

  function attachContext(context) {
    if (!context) return;
    context = structuredClone(context);
    if (!state.privacy.include_run) {
      context.selection.run_id = null;
      context.selection.event_id = null;
      context.display.run = null;
      context.display.event = null;
    }
    state.context = context;
    const chips = select('#assistant-context');
    chips.replaceChildren();
    for (const part of context.display.parts) chips.append(element('span', 'context-chip', part.name));
    if (!context.display.parts.length) chips.append(element('span', 'context-chip', context.display.scene));
    if (context.display.run) chips.append(element('span', 'context-chip run', context.display.run));
    if (context.display.event) chips.append(element('span', 'context-chip event', context.display.event));
    if (context.selection.selected_text) chips.append(element('span', 'context-chip highlight', 'Highlighted text'));
    const details = [
      `Assembly: ${context.display.scene}`,
      `Parts: ${context.display.parts.map(part => part.name).join(', ') || 'Assembly overview'}`,
      state.privacy.include_source ? 'Server adds the selected parts’ catalog descriptions, bounded source excerpts, and immediate connections.' : 'Catalog descriptions and connections are included. Automatic source excerpts are disabled in Settings.',
      'The visual workspace layout, connections, and notes are included as draft context.',
      context.display.run ? `Recorded run summary: ${context.display.run}` : 'No recorded run attached.',
      context.display.event ? `Recorded event: ${context.display.event}` : '',
      context.selection.selected_text ? `Highlighted text:\n${context.selection.selected_text}` : '',
      'Earlier messages in this conversation are also included. No shell, browser, or source-editing tools are available.',
    ];
    select('#assistant-context-preview').textContent = details.filter(Boolean).join('\n\n');
  }

  function operationLabel(operation) {
    const names = {move_component: 'Move', set_note: 'Note', connect: 'Connect', disconnect: 'Disconnect', add_component: 'Add visual part', remove_component: 'Remove visual part'};
    if (operation.type === 'connect' || operation.type === 'disconnect') return `${names[operation.type]}: ${operation.source} → ${operation.target}${operation.label ? ` · ${operation.label}` : ''}`;
    if (operation.type === 'set_note') return `Note · ${operation.component_id}: ${operation.text}`;
    return `${names[operation.type] || operation.type} · ${operation.name || operation.component_id}${operation.x !== undefined ? ` → (${operation.x}, ${operation.y})` : ''}`;
  }

  function renderMessages() {
    const log = select('#assistant-messages');
    const welcome = select('#assistant-welcome');
    log.querySelectorAll('.chat-message').forEach(node => node.remove());
    welcome.hidden = state.messages.length > 0;
    for (const message of state.messages) {
      const article = element('article', `chat-message ${message.role}`);
      article.append(element('span', 'chat-role', message.role === 'user' ? 'YOU' : message.role === 'system' ? 'WORKSPACE' : (message.label || 'ChatGPT Sub').toUpperCase()));
      article.append(element('div', 'chat-text', message.text));
      if (message.context) article.append(element('small', 'chat-context-label', message.context));
      if (message.proposal && !message.dismissed) {
        const card = element('section', 'proposal-card');
        card.append(element('span', 'draft-badge', 'VISUAL DRAFT'));
        card.append(element('h3', '', message.proposal.title));
        card.append(element('p', '', message.proposal.description));
        const operations = element('ul', 'proposal-operations');
        for (const operation of message.proposal.operations) operations.append(element('li', 'proposal-operation', operationLabel(operation)));
        card.append(operations);
        const actions = element('div', 'proposal-actions');
        const preview = element('button', 'primary-button', 'Preview changes');
        preview.type = 'button';
        preview.addEventListener('click', () => {
          try {
            bridge().preview(message.proposal, message.scene);
            showError();
            const notice = select('#assistant-draft-status');
            notice.hidden = false;
            notice.textContent = 'Preview is on the canvas. Close this panel to inspect it, then Apply draft or Dismiss preview. Your source stays unchanged.';
          } catch (error) { showError(error.message); }
        });
        const dismiss = element('button', 'secondary-button', 'Dismiss');
        dismiss.type = 'button';
        dismiss.addEventListener('click', () => { message.dismissed = true; bridge().dismissPreview(); persist(); renderMessages(); });
        actions.append(preview, dismiss);
        card.append(actions);
        article.append(card);
      }
      log.append(article);
    }
    log.scrollTop = log.scrollHeight;
  }

  function appendMessage(message) {
    state.messages.push(message);
    state.messages = state.messages.slice(-30);
    persist();
    renderMessages();
  }

  async function pollJob() {
    if (state.polling || !state.job) return;
    state.polling = true;
    setBusy(true);
    try {
      while (state.job) {
        const pending = state.job;
        const result = await request(`/api/assistant/jobs/${encodeURIComponent(pending.id)}`);
        if (result.status === 'running' || result.status === 'queued') {
          await new Promise(resolve => setTimeout(resolve, 850));
          continue;
        }
        if (result.status === 'completed') {
          appendMessage({role: 'assistant', text: result.answer || 'The visual proposal is ready to review.', proposal: result.proposal, scene: pending.scene, label: result.label || state.label});
          if (result.model) select('#assistant-model').textContent = result.model;
        } else {
          const fallback = result.status === 'cancelled' ? 'Request stopped. Your workspace is unchanged.' : result.status === 'uncertain' ? 'The server restarted during this request. It was not automatically sent again.' : 'The assistant request failed. No workspace changes were applied.';
          appendMessage({role: 'system', text: result.error || fallback});
        }
        state.job = null;
        persist();
        setBusy(false);
      }
    } catch (error) {
      if (error.status === 404) {
        state.job = null;
        persist();
        setBusy(false);
        appendMessage({role: 'system', text: 'The saved request is no longer available on this server. It was not sent again. You can start a new message.'});
        showError();
      } else showError(`Connection interrupted: ${error.message} Close and reopen chat to check this request. It will not be sent again automatically.`);
    } finally {
      state.polling = false;
    }
  }

  async function sendPending(refresh = false) {
    if (!state.pending || state.submitting) return;
    const pending = state.pending;
    state.submitting = true;
    setBusy(true);
    showError();
    try {
      if (refresh) await checkConnection();
      if (!state.csrf) throw new Error('Could not refresh the local connection. Reopen chat or try Recover request again.');
      const result = await request('/api/assistant/messages', pending.body);
      if (!identifier(result?.job_id) || !identifier(result?.conversation_id)) throw new Error('The server did not confirm a valid request ID.');
      state.conversationId = result.conversation_id;
      state.job = {id: result.job_id, scene: pending.scene};
      state.pending = null;
      persist();
      showError();
    } catch (error) {
      if (error.status === 400 || error.status === 413) {
        state.pending = null;
        persist();
        const input = select('#assistant-input');
        if (!input.value) input.value = pending.body.message;
        appendMessage({role: 'system', text: `Request rejected: ${error.message} Update your message or selection and send again.`});
      } else {
        showError(`Unable to confirm the request: ${error.message} Use Recover request to check with the same message and ID. Nothing is retried automatically.`);
      }
    } finally {
      state.submitting = false;
      setBusy(Boolean(state.job || state.pending));
    }
    if (state.job) pollJob();
  }

  async function open(context) {
    if (panel.hidden) state.returnFocus = document.activeElement;
    panel.hidden = false;
    select('#assistant-open').setAttribute('aria-expanded', 'true');
    attachContext(context || bridge()?.context());
    renderMessages();
    select('#assistant-input').focus();
    await checkConnection();
    if (state.job) pollJob();
  }

  function close() {
    panel.hidden = true;
    select('#assistant-open').setAttribute('aria-expanded', 'false');
    if (state.returnFocus?.isConnected) state.returnFocus.focus();
  }

  try {
    const serialized = sessionStorage.getItem(storageKey) || 'null';
    const saved = serialized.length <= 1024 * 1024 ? JSON.parse(serialized) : null;
    if (record(saved)) {
      state.messages = Array.isArray(saved.messages) ? saved.messages.slice(-30).map(savedMessage).filter(Boolean) : [];
      state.conversationId = identifier(saved.conversationId) ? saved.conversationId : null;
      state.job = record(saved.job) && identifier(saved.job.id) && sceneId(saved.job.scene) ? {id: saved.job.id, scene: saved.job.scene} : null;
      state.pending = !state.job && validPending(saved.pending) ? {body: saved.pending.body, scene: saved.pending.scene} : null;
      state.busy = Boolean(state.job || state.pending);
      if (textWithin(saved.routing, 600)) state.routing = saved.routing;
    }
  } catch {}

  recover.addEventListener('click', () => sendPending(true));
  select('#assistant-open').addEventListener('click', () => open());
  select('#assistant-close').addEventListener('click', close);
  select('#assistant-selection-update').addEventListener('click', () => attachContext(bridge()?.context(window.getSelection()?.toString() || '')));
  select('#assistant-new').addEventListener('click', () => {
    if (state.busy) return;
    state.messages = [];
    state.conversationId = null;
    select('#assistant-draft-status').hidden = true;
    persist();
    renderMessages();
    attachContext(bridge()?.context());
    select('#assistant-input').focus();
  });
  select('#assistant-stop').addEventListener('click', async () => {
    if (!state.job) return;
    select('#assistant-stop').disabled = true;
    try {
      await request(`/api/assistant/jobs/${encodeURIComponent(state.job.id)}/cancel`, {});
      if (!state.polling) pollJob();
    } catch (error) { showError(error.message); }
    finally { select('#assistant-stop').disabled = false; }
  });
  select('#assistant-form').addEventListener('submit', async event => {
    event.preventDefault();
    const input = select('#assistant-input');
    const message = input.value.trim();
    if (!message || state.busy || !state.available || !state.context || bridge()?.hasPreview?.()) return;
    const context = structuredClone(state.context);
    state.pending = {body: {request_id: crypto.randomUUID(), conversation_id: state.conversationId, message, selection: context.selection, workspace: context.workspace}, scene: context.selection.scene};
    if (!persist()) {
      state.pending = null;
      showError('Browser session storage is unavailable. Enable it before sending so an interrupted request can be recovered safely.');
      return;
    }
    setBusy(true);
    showError();
    appendMessage({role: 'user', text: message, context: context.display.parts.map(part => part.name).join(' · ') || context.display.scene});
    input.value = '';
    await sendPending();
  });
  select('#assistant-input').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); select('#assistant-form').requestSubmit(); }
  });
  panel.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
  });
  document.querySelectorAll('[data-chat-prompt]').forEach(button => button.addEventListener('click', () => {
    const prompts = {explain: 'Explain the selected parts and their responsibilities. How do they fit into the harness?', trace: 'Trace how data moves through the selected parts and their immediate connections.', improve: 'Suggest a clearer workflow for these parts as a visual draft. Explain the tradeoffs and provide changes I can preview. Do not change source files.'};
    select('#assistant-input').value = prompts[button.dataset.chatPrompt];
    select('#assistant-input').focus();
  }));
  window.addEventListener('workbench:ask', event => {
    open(event.detail.context);
    if (typeof event.detail.prompt === 'string' && event.detail.prompt) select('#assistant-input').value = event.detail.prompt.slice(0, 6000);
  });
  window.addEventListener('workbench:settings-saved', event => {
    const settings = event.detail.settings;
    updateProvider({provider: settings.provider, configured_model: settings[settings.provider]?.model || '', privacy: settings.privacy});
    if (!panel.hidden) checkConnection();
  });
  window.addEventListener('workbench:selection', () => { if (!panel.hidden) attachContext(bridge()?.context()); setBusy(state.busy); });
  window.addEventListener('workbench:draft-applied', event => {
    select('#assistant-draft-status').hidden = false;
    select('#assistant-draft-status').textContent = `${event.detail.title} applied to this visual workspace. Use Undo to revert. Harness source is unchanged.`;
    attachContext(bridge()?.context());
  });
  setBusy(state.busy);
  window.hyperspaceAssistant = {busy: () => state.busy};
})();
