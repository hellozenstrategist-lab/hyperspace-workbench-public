(() => {
  'use strict';

  const find = (identifier) => document.getElementById(identifier);
  const state = {settings: null, csrf: null, selectedProvider: 'chatgpt', loading: false, saving: false, dirty: false, modelLoads: new Set(), checkingStatus: false};
  const preferenceControls = {
    snap_to_grid: 'settings-snap-to-grid',
    show_guides: 'settings-show-guides',
    decorative_art: 'settings-decorative-art',
    reduced_motion: 'settings-reduced-motion',
    auto_refresh: 'settings-auto-refresh',
    refresh_interval: 'settings-refresh-interval',
  };

  function assistantBusy() {
    return Boolean(window.hyperspaceWorkbench?.assistantBusy?.());
  }

  function showError(message) {
    const output = find('settings-ai-error');
    output.textContent = message;
    output.hidden = !message;
  }

  function setStatus(message, kind = '') {
    const output = find('settings-ai-status');
    output.textContent = message;
    output.className = 'settings-save-status' + (kind ? ' ' + kind : '');
  }

  function updateLock() {
    const busy = assistantBusy();
    const locked = busy || state.loading || state.saving || !state.settings;
    find('settings-busy-notice').hidden = !busy;
    find('settings-ai-fields').disabled = locked;
    find('settings-ai-save').disabled = locked;
    find('settings-ai-save').textContent = state.saving ? 'Saving…' : 'Save AI settings';
    for (const provider of ['chatgpt', 'openrouter']) {
      find(`settings-${provider}-load-models`).disabled = locked || state.modelLoads.has(provider);
    }
    find('settings-chatgpt-status-button').disabled = locked || state.checkingStatus;
    find('settings-openrouter-secret').disabled = locked || find('settings-clear-secret').checked;
    find('settings-remember-secret').disabled = locked || find('settings-clear-secret').checked;
  }

  function selectProvider(provider) {
    state.selectedProvider = provider;
    find('settings-provider-chatgpt').checked = provider === 'chatgpt';
    find('settings-provider-openrouter').checked = provider === 'openrouter';
    find('settings-chatgpt-fields').hidden = provider !== 'chatgpt';
    find('settings-openrouter-fields').hidden = provider !== 'openrouter';
  }

  function publicSettings(value) {
    if (!value || !['chatgpt', 'openrouter'].includes(value.provider)) throw new Error('The local reader returned an invalid settings response.');
    return {
      provider: value.provider,
      chatgpt: {model: String(value.chatgpt?.model || '')},
      openrouter: {
        model: String(value.openrouter?.model || ''),
        max_tokens: Number.isInteger(value.openrouter?.max_tokens) ? value.openrouter.max_tokens : 2048,
        credential_present: value.openrouter?.credential_present === true,
        credential_source: ['session', 'saved', 'environment', 'none'].includes(value.openrouter?.credential_source) ? value.openrouter.credential_source : 'none',
      },
      privacy: {include_source: value.privacy?.include_source !== false, include_run: value.privacy?.include_run !== false},
    };
  }

  function populateSettings(payload) {
    const value = payload.settings || payload;
    state.csrf = payload.csrf || value.csrf || state.csrf;
    state.settings = publicSettings(value);
    const settings = state.settings;
    selectProvider(settings.provider);
    find('settings-chatgpt-model').value = settings.chatgpt.model;
    find('settings-openrouter-model').value = settings.openrouter.model;
    find('settings-openrouter-max-tokens').value = settings.openrouter.max_tokens;
    find('settings-include-source').checked = settings.privacy.include_source;
    find('settings-include-run').checked = settings.privacy.include_run;
    find('settings-openrouter-secret').value = '';
    find('settings-remember-secret').checked = false;
    find('settings-clear-secret').checked = false;
    const labels = {session: 'Key available for this server session', saved: 'Key saved on this machine', environment: 'Key supplied by environment', none: 'No key configured'};
    find('settings-key-status').textContent = settings.openrouter.credential_present ? labels[settings.openrouter.credential_source] || 'Key available' : 'No key configured';
    state.dirty = false;
  }

  async function request(path, body) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? 'GET' : 'POST',
        headers: body === undefined ? {} : {'Content-Type': 'application/json', 'X-Workbench-CSRF': state.csrf},
        credentials: 'same-origin', cache: 'no-store', signal: controller.signal,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.error === 'string' ? payload.error : `Settings request failed (${response.status}).`);
      return payload;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function loadSettings() {
    updateLock();
    await renderPreferences();
    if (state.loading || state.saving || state.dirty) return;
    state.loading = true;
    showError('');
    find('settings-loading').textContent = 'Reading saved settings…';
    updateLock();
    try {
      const payload = await request('/api/assistant/settings');
      populateSettings(payload);
      find('settings-loading').textContent = '';
      if (payload.error) showError(String(payload.error));
    } catch (error) {
      find('settings-loading').textContent = 'Settings unavailable. Open Settings again to retry.';
      showError(error.name === 'AbortError' ? 'The local reader timed out while loading settings.' : error.message);
    } finally {
      state.loading = false;
      updateLock();
    }
  }

  function modelValue(identifier) {
    const model = find(identifier).value.trim();
    if (model.length > 200 || (model && !/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/.test(model)) || model.includes('://')) {
      throw new Error('Enter an exact model ID using letters, numbers, slashes, dots, hyphens, underscores, or colons.');
    }
    return model;
  }

  function makePayload() {
    const provider = state.selectedProvider;
    const chatgptModel = modelValue('settings-chatgpt-model');
    const openrouterModel = modelValue('settings-openrouter-model');
    if (provider === 'openrouter' && !openrouterModel) throw new Error('Enter an explicit OpenRouter model ID.');
    if (['auto', 'openrouter/auto', 'openrouter/free'].includes(openrouterModel.toLowerCase()) || openrouterModel.toLowerCase().includes(':online')) {
      throw new Error('Choose an explicit model. Automatic routing and web-enabled models are unavailable.');
    }
    const maxTokens = Number(find('settings-openrouter-max-tokens').value);
    if (!Number.isInteger(maxTokens) || maxTokens < 256 || maxTokens > 8192) throw new Error('Maximum output tokens must be a whole number from 256 to 8192.');
    const openrouter = {model: openrouterModel, max_tokens: maxTokens};
    const secret = find('settings-openrouter-secret').value.trim();
    const remember = find('settings-remember-secret').checked;
    const clear = find('settings-clear-secret').checked;
    if (clear) {
      openrouter.clear_secret = true;
    } else if (secret) {
      if (secret.length > 4096 || /\s/.test(secret)) throw new Error('The API key must contain no whitespace and be at most 4096 characters.');
      openrouter.secret = secret;
      openrouter.remember_secret = remember;
    } else if (remember) {
      throw new Error('Paste a new API key before choosing Remember on this machine.');
    }
    return {
      provider,
      chatgpt: {model: chatgptModel},
      openrouter,
      privacy: {include_source: find('settings-include-source').checked, include_run: find('settings-include-run').checked},
    };
  }

  async function saveSettings(event) {
    event.preventDefault();
    if (assistantBusy()) { updateLock(); return; }
    if (state.loading || state.saving || !state.settings) return;
    showError('');
    let payload;
    try {
      if (!state.csrf) throw new Error('Reopen Settings to refresh the local connection before saving.');
      payload = makePayload();
    } catch (error) {
      showError(error.message);
      return;
    }
    state.saving = true;
    setStatus('Saving AI settings…');
    updateLock();
    try {
      const result = await request('/api/assistant/settings', payload);
      populateSettings(result);
      setStatus('AI settings saved.', 'saved');
      window.dispatchEvent(new CustomEvent('workbench:settings-saved', {detail: {settings: state.settings}}));
    } catch (error) {
      state.dirty = false;
      setStatus('AI settings were not confirmed.', 'error');
      showError(error.name === 'AbortError' ? 'The save could not be confirmed. No automatic retry was made. Reopen Settings to check the saved connection.' : error.message);
    } finally {
      delete payload.openrouter.secret;
      find('settings-openrouter-secret').value = '';
      find('settings-remember-secret').checked = false;
      find('settings-clear-secret').checked = false;
      state.saving = false;
      updateLock();
    }
  }

  async function loadModels(provider) {
    if (assistantBusy() || state.modelLoads.has(provider)) { updateLock(); return; }
    state.modelLoads.add(provider);
    showError('');
    setStatus('Loading model choices…');
    updateLock();
    try {
      const payload = await request(`/api/assistant/models/${provider}`);
      if (!Array.isArray(payload.models)) throw new Error('The local reader did not return a model list.');
      const list = find(`settings-${provider}-model-list`);
      list.replaceChildren();
      let count = 0;
      for (const model of payload.models) {
        if (!model || typeof model.id !== 'string') continue;
        const option = document.createElement('option');
        option.value = model.id;
        option.label = typeof model.name === 'string' ? model.name : model.id;
        list.append(option);
        count += 1;
      }
      setStatus(count ? `${count} model choices loaded. Type in Model to filter them.` : 'No model choices returned. You can enter an exact model ID.');
      find(`settings-${provider}-model`).focus();
    } catch (error) {
      setStatus('Model list unavailable.', 'error');
      showError(error.name === 'AbortError' ? 'The model list request timed out. You can still enter an exact model ID.' : error.message);
    } finally {
      state.modelLoads.delete(provider);
      updateLock();
    }
  }

  async function checkConnection() {
    if (assistantBusy() || state.checkingStatus) { updateLock(); return; }
    state.checkingStatus = true;
    find('settings-chatgpt-status').textContent = 'Checking the saved connection…';
    updateLock();
    try {
      const status = await request('/api/assistant/status');
      const provider = status.label || (status.provider === 'openrouter' ? 'OpenRouter' : 'ChatGPT Sub');
      find('settings-chatgpt-status').textContent = status.available ? `${provider} is available. No reply was generated.` : `${provider}: ${status.error || (status.authenticated ? 'Connection unavailable.' : 'No active login or credential.')}`;
    } catch (error) {
      find('settings-chatgpt-status').textContent = error.name === 'AbortError' ? 'Connection status timed out.' : error.message;
    } finally {
      state.checkingStatus = false;
      updateLock();
    }
  }

  async function renderPreferences() {
    const preferences = window.hyperspacePreferences;
    const output = find('settings-preferences-status');
    if (!preferences?.get) { output.textContent = 'Workspace preferences are unavailable.'; return; }
    try {
      const values = await preferences.get();
      for (const [name, identifier] of Object.entries(preferenceControls)) {
        if (name === 'refresh_interval') find(identifier).value = String(values[name]);
        else find(identifier).checked = values[name] === true;
      }
    } catch {
      output.textContent = 'Could not read workspace preferences.';
      output.className = 'settings-save-status error';
    }
  }

  async function savePreference(name, control) {
    const output = find('settings-preferences-status');
    try {
      if (!window.hyperspacePreferences?.set) throw new Error('Workspace preferences are unavailable.');
      const value = name === 'refresh_interval' ? Number(control.value) : control.checked;
      await window.hyperspacePreferences.set({[name]: value});
      output.textContent = 'Preference saved in this browser.';
      output.className = 'settings-save-status saved';
    } catch (error) {
      output.textContent = error.message || 'Could not save this preference.';
      output.className = 'settings-save-status error';
      await renderPreferences();
    }
  }

  async function resetPreferences() {
    const output = find('settings-preferences-status');
    try {
      if (!window.hyperspacePreferences?.reset) throw new Error('Workspace preferences are unavailable.');
      await window.hyperspacePreferences.reset();
      await renderPreferences();
      output.textContent = 'Workspace preferences reset.';
      output.className = 'settings-save-status saved';
    } catch (error) {
      output.textContent = error.message || 'Could not reset workspace preferences.';
      output.className = 'settings-save-status error';
    }
  }

  function initialize() {
    if (!find('settings-view')) return;
    find('settings-ai-form').addEventListener('submit', saveSettings);
    find('settings-ai-form').addEventListener('input', () => {
      if (state.saving || state.loading) return;
      state.dirty = true;
      setStatus('Unsaved AI settings.');
    });
    for (const provider of ['chatgpt', 'openrouter']) {
      find(`settings-provider-${provider}`).addEventListener('change', () => {
        if (assistantBusy()) { selectProvider(state.selectedProvider); updateLock(); return; }
        selectProvider(provider);
        state.dirty = true;
        setStatus('Provider change applies when you save.');
      });
      find(`settings-${provider}-load-models`).addEventListener('click', () => loadModels(provider));
    }
    find('settings-chatgpt-status-button').addEventListener('click', checkConnection);
    find('settings-clear-secret').addEventListener('change', () => {
      if (find('settings-clear-secret').checked) {
        find('settings-openrouter-secret').value = '';
        find('settings-remember-secret').checked = false;
      }
      updateLock();
    });
    for (const [name, identifier] of Object.entries(preferenceControls)) {
      find(identifier).addEventListener('change', (event) => savePreference(name, event.currentTarget));
    }
    find('settings-preferences-reset').addEventListener('click', resetPreferences);
    window.addEventListener('workbench:view', (event) => { if (event.detail?.view === 'settings') loadSettings(); });
    window.addEventListener('workbench:preferences-changed', renderPreferences);
    window.setInterval(() => { if (!find('settings-view').hidden) updateLock(); }, 400);
    if (!find('settings-view').hidden) loadSettings();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, {once: true});
  else initialize();
})();
