const {chromium} = await import(process.env.WORKBENCH_PLAYWRIGHT_MODULE || 'playwright');
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';

const base = process.env.WORKBENCH_URL || 'http://127.0.0.1:8765/';
if (!['127.0.0.1', 'localhost'].includes(new URL(base).hostname)) throw new Error('Only local Workbench verification is supported.');
const directory = process.env.WORKBENCH_BROWSER_ARTIFACTS || '/tmp/hyperspace-workbench-qa';
await mkdir(directory, {recursive: true});
const browser = await chromium.launch({channel: 'chrome', headless: true});
const page = await browser.newPage({viewport: {width: 1600, height: 1000}});
const checks = [];
const errors = [];
const messages = [];
const settingsWrites = [];
const modelLoads = [];
const secret = 'sk-or-v1-synthetic-browser-settings-only';
const preferencesDefaults = {snap_to_grid: true, show_guides: true, decorative_art: true, reduced_motion: false, auto_refresh: false, refresh_interval: 15};
let settings = {provider: 'chatgpt', chatgpt: {model: ''}, openrouter: {model: '', max_tokens: 2048, credential_present: false, credential_source: 'none'}, privacy: {include_source: true, include_run: true}, csrf: 'offline-settings-nonce'};
const fulfill = (route, data, status = 200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(data)});
const check = name => {checks.push(name); console.log('PASS ' + name);};
const storage = () => page.evaluate(() => JSON.stringify({local: {...localStorage}, session: {...sessionStorage}}));
const workspace = () => page.evaluate(() => JSON.parse(localStorage.getItem('hyperspace-workbench-v1') || 'null'));
const waitForSettings = () => page.waitForFunction(() => !document.querySelector('#settings-ai-save').disabled);
const openSettings = async () => {
  await page.locator('[data-view="settings"]').click();
  await waitForSettings();
};
const saveSettings = async () => {
  const count = settingsWrites.length;
  await page.locator('#settings-ai-save').click();
  await page.waitForFunction(() => document.querySelector('#settings-ai-status').textContent === 'AI settings saved.');
  await waitForSettings();
  assert.equal(settingsWrites.length, count + 1);
};
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
await page.route('**/api/assistant/**', route => {
  const request = route.request();
  const pathname = new URL(request.url()).pathname;
  if (pathname === '/api/assistant/settings') {
    if (request.method() === 'POST') {
      assert.equal(request.headers()['x-workbench-csrf'], settings.csrf);
      const body = request.postDataJSON();
      settingsWrites.push(structuredClone(body));
      settings.provider = body.provider;
      settings.chatgpt = {...settings.chatgpt, ...body.chatgpt};
      settings.privacy = {...settings.privacy, ...body.privacy};
      const incoming = body.openrouter || {};
      if (incoming.secret) {
        settings.openrouter.credential_present = true;
        settings.openrouter.credential_source = incoming.remember_secret ? 'saved' : 'session';
      }
      if (incoming.clear_secret) {
        settings.openrouter.credential_present = false;
        settings.openrouter.credential_source = 'none';
      }
      for (const field of ['model', 'max_tokens']) if (Object.hasOwn(incoming, field)) settings.openrouter[field] = incoming[field];
    }
    assert.equal(JSON.stringify(settings).includes(secret), false);
    return fulfill(route, settings);
  }
  if (pathname === '/api/assistant/status') {
    const native = settings.provider === 'chatgpt';
    const available = native || Boolean(settings.openrouter.model && settings.openrouter.credential_present);
    return fulfill(route, {authenticated: available, available, provider: settings.provider, label: native ? 'ChatGPT Sub' : 'OpenRouter', model: settings[settings.provider].model || 'Account default', configured_model: settings[settings.provider].model, cli_version: '0.154.0-offline', privacy: settings.privacy, csrf: settings.csrf});
  }
  if (pathname.startsWith('/api/assistant/models/')) {
    const provider = pathname.split('/').at(-1);
    modelLoads.push(provider);
    return fulfill(route, {provider, models: [{id: `example/${provider}`, name: `Offline ${provider}`}], default: provider === 'chatgpt' ? 'example/chatgpt' : ''});
  }
  if (pathname === '/api/assistant/messages') {
    messages.push(request.postDataJSON());
    return fulfill(route, {error: 'The settings browser suite never requests generation.'}, 409);
  }
  if (pathname === '/api/assistant/history') return fulfill(route, {conversations: []});
  return fulfill(route, {error: 'Unexpected assistant endpoint in settings verification.'}, 404);
});
await page.route('**/api/components/coordinator/source', route => fulfill(route, {path: 'astra_harness/coordinator.py', content: 'class Coordinator:\n    responsibility = "Synthetic source for UI inspection"\n', truncated: false}));

try {
  await page.goto(base);
  await page.waitForSelector('[data-node="coordinator"]');
  await openSettings();
  assert.equal(await page.locator('#settings-view').isVisible(), true);
  assert.equal(await page.locator('#settings-provider-chatgpt').isChecked(), true);
  assert.equal(await page.locator('#settings-chatgpt-model').inputValue(), '');
  assert.deepEqual(modelLoads, []);
  assert.deepEqual(messages, []);
  await page.locator('#settings-chatgpt-load-models').click();
  await page.locator('#settings-chatgpt-model-list option').waitFor({state: 'attached'});
  assert.equal(await page.locator('#settings-chatgpt-model-list option').getAttribute('value'), 'example/chatgpt');
  await page.locator('#settings-chatgpt-status-button').click();
  await page.waitForFunction(() => document.querySelector('#settings-chatgpt-status').textContent.includes('No reply was generated'));
  assert.deepEqual(modelLoads, ['chatgpt']);
  assert.deepEqual(messages, []);
  check('Settings default to ChatGPT; explicit status and model listing never generate');

  await page.locator('#settings-provider-openrouter').check();
  await page.locator('#settings-openrouter-load-models').click();
  await page.locator('#settings-openrouter-model-list option').waitFor({state: 'attached'});
  await page.locator('#settings-openrouter-model').fill('example/openrouter');
  await page.locator('#settings-openrouter-max-tokens').fill('4096');
  await page.locator('#settings-openrouter-secret').fill(secret);
  assert.equal(await page.locator('#settings-remember-secret').isChecked(), false);
  await page.locator('#settings-include-source').uncheck();
  await page.locator('#settings-include-run').uncheck();
  await saveSettings();
  assert.equal(settings.provider, 'openrouter');
  assert.equal(settingsWrites.at(-1).openrouter.secret, secret);
  assert.equal(settingsWrites.at(-1).openrouter.remember_secret, false);
  assert.equal(await page.locator('#settings-openrouter-secret').inputValue(), '');
  assert.match(await page.locator('#settings-key-status').innerText(), /server session/);
  assert.equal((await storage()).includes(secret), false);
  await page.locator('#assistant-open').click();
  await page.waitForFunction(() => document.querySelector('#assistant-title').textContent === 'OpenRouter' && !document.querySelector('#assistant-send').disabled);
  assert.equal(await page.locator('#assistant-model').textContent(), 'example/openrouter');
  assert.match(await page.locator('.assistant-privacy').textContent(), /Separate provider charges/);
  await page.locator('#assistant-close').click();
  assert.deepEqual(messages, []);
  check('Explicit OpenRouter choice updates labels and privacy; session key never enters browser storage');

  await page.reload();
  await page.waitForSelector('[data-node="coordinator"]');
  await openSettings();
  assert.equal(await page.locator('#settings-provider-openrouter').isChecked(), true);
  assert.equal(await page.locator('#settings-openrouter-model').inputValue(), 'example/openrouter');
  assert.equal(await page.locator('#settings-openrouter-max-tokens').inputValue(), '4096');
  assert.equal(await page.locator('#settings-include-source').isChecked(), false);
  assert.equal(await page.locator('#settings-openrouter-secret').inputValue(), '');
  assert.deepEqual(modelLoads, ['chatgpt', 'openrouter']);
  await page.locator('#settings-clear-secret').check();
  await saveSettings();
  assert.equal(settingsWrites.at(-1).openrouter.clear_secret, true);
  assert.equal(Object.hasOwn(settingsWrites.at(-1).openrouter, 'secret'), false);
  assert.equal(settings.openrouter.credential_present, false);
  assert.equal(await page.locator('#settings-key-status').innerText(), 'No key configured');
  assert.equal((await storage()).includes(secret), false);
  check('Saved provider reloads without model calls; clearing the key is explicit and write-only');

  const beforeInvalid = structuredClone(settings);
  const writesBeforeInvalid = settingsWrites.length;
  await page.locator('#settings-openrouter-model').fill('openrouter/auto');
  await page.locator('#settings-ai-save').click();
  assert.equal(await page.locator('#settings-ai-error').isVisible(), true);
  assert.match(await page.locator('#settings-ai-error').textContent(), /explicit model/);
  assert.equal(settingsWrites.length, writesBeforeInvalid);
  assert.deepEqual(settings, beforeInvalid);
  await page.locator('#settings-openrouter-model').fill('example/openrouter');
  await page.locator('#settings-openrouter-max-tokens').fill('255');
  await page.locator('#settings-ai-save').click();
  assert.equal(await page.locator('#settings-openrouter-max-tokens').evaluate(input => input.validity.rangeUnderflow), true);
  assert.equal(settingsWrites.length, writesBeforeInvalid);
  assert.deepEqual(settings, beforeInvalid);
  await page.locator('#settings-openrouter-max-tokens').fill('2048');
  await page.locator('#settings-provider-chatgpt').check();
  await saveSettings();
  check('Invalid automatic models and output bounds preserve configuration without a POST');

  for (const identifier of ['settings-snap-to-grid', 'settings-show-guides', 'settings-decorative-art']) await page.locator('#' + identifier).uncheck();
  await page.locator('#settings-reduced-motion').check();
  await page.locator('#settings-auto-refresh').check();
  await page.locator('#settings-refresh-interval').selectOption('60');
  const changedPreferences = {snap_to_grid: false, show_guides: false, decorative_art: false, reduced_motion: true, auto_refresh: true, refresh_interval: 60};
  assert.deepEqual(await page.evaluate(() => window.hyperspacePreferences.get()), changedPreferences);
  assert.equal(await page.locator('body').evaluate(body => body.classList.contains('prefs-no-art') && body.classList.contains('prefs-reduced-motion')), true);
  assert.equal(await page.locator('.canvas-guidebar').evaluate(node => node.hidden), true);
  assert.equal(await page.locator('#live-toggle').isChecked(), true);
  await page.reload();
  await page.waitForSelector('[data-node="coordinator"]');
  await openSettings();
  assert.deepEqual(await page.evaluate(() => window.hyperspacePreferences.get()), changedPreferences);
  assert.equal(await page.locator('#settings-snap-to-grid').isChecked(), false);
  assert.equal(await page.locator('#settings-refresh-interval').inputValue(), '60');
  await page.locator('#settings-preferences-reset').click();
  assert.deepEqual(await page.evaluate(() => window.hyperspacePreferences.get()), preferencesDefaults);
  assert.equal(await page.locator('.canvas-guidebar').evaluate(node => node.hidden), false);
  assert.equal(await page.locator('#live-toggle').isChecked(), false);
  assert.equal(await page.locator('body').evaluate(body => body.classList.contains('prefs-no-art') || body.classList.contains('prefs-reduced-motion')), false);
  assert.deepEqual(messages, []);
  await page.screenshot({path: path.join(directory, 'settings-provider-preferences.png')});
  check('Snap, guides, art, motion and refresh preferences persist, affect the UI, and reset');

  await page.locator('[data-view="assembly"]').click();
  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-ai').click();
  await page.locator('#node-ai-menu').waitFor();
  for (const identifier of ['ask', 'inspect', 'refactor', 'run', 'explain', 'worker']) assert.equal(await page.locator('#node-ai-' + identifier).isVisible(), true);
  await page.locator('#node-ai-run').click();
  await page.locator('#assistant-panel').waitFor();
  assert.match(await page.locator('#assistant-input').inputValue(), /simulation/);
  assert.match(await page.locator('#assistant-input').inputValue(), /do not execute/);
  assert.deepEqual(messages, []);
  await page.locator('#assistant-close').click();
  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-ai').click();
  await page.locator('#node-ai-inspect').click();
  await page.locator('#inspector-content .source-listing').waitFor();
  assert.match(await page.locator('#inspector-content .code-view').textContent(), /Synthetic source/);
  assert.deepEqual(messages, []);
  check('Call AI submenu exposes bounded actions; Run only prefills a simulation and Inspect reads source');

  await page.locator('[data-inspector-tab="ai"]').click();
  const beforeProfile = await workspace();
  await page.locator('#part-ai-enabled').uncheck();
  await page.locator('#part-ai-primary-model').fill('example/visual-primary');
  await page.locator('#part-ai-fallback-model').fill('example/visual-fallback');
  await page.locator('#part-ai-max-steps').fill('12');
  await page.locator('#part-ai-temperature').fill('0.4');
  await page.locator('#part-ai-preview').click();
  assert.equal(await page.locator('#draft-preview-bar').isVisible(), true);
  assert.deepEqual(await workspace(), beforeProfile);
  assert.deepEqual(messages, []);
  await page.locator('#draft-apply').click();
  const appliedProfile = await workspace();
  assert.deepEqual(appliedProfile.ai_parts.coordinator, {enabled: false, primary_model: 'example/visual-primary', fallback_model: 'example/visual-fallback', max_steps: 12, temperature: 0.4});
  assert.equal(settings.provider, 'chatgpt');
  assert.equal(settings.chatgpt.model, '');
  await page.locator('#undo-button').click();
  const undoneProfile = await workspace();
  assert.deepEqual(undoneProfile?.ai_parts || {}, beforeProfile?.ai_parts || {});
  assert.deepEqual(undoneProfile?.drafts || {}, beforeProfile?.drafts || {});
  assert.deepEqual(messages, []);
  check('Part AI profile previews without persistence, applies only to the draft, and undoes without dispatch');

  const tabPositions = await page.locator('.inspector-tabs button').evaluateAll(buttons => buttons.map(button => button.getBoundingClientRect().top));
  assert.equal(new Set(tabPositions).size, 1);
  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-ai').click();
  await page.screenshot({path: path.join(directory, 'call-ai-submenu.png')});
  assert.equal(await page.locator('#node-menu-ai').getAttribute('aria-expanded'), 'true');
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#node-menu-ai').getAttribute('aria-expanded'), 'false');
  await page.keyboard.press('Escape');
  await page.setViewportSize({width: 1024, height: 768});
  await page.locator('#fit-view').click();
  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-ai').click();
  const submenuBounds = await page.locator('#node-ai-menu').boundingBox();
  assert.ok(submenuBounds.x >= 0 && submenuBounds.y >= 0);
  assert.ok(submenuBounds.x + submenuBounds.width <= 1024 && submenuBounds.y + submenuBounds.height <= 768);
  await page.screenshot({path: path.join(directory, 'call-ai-compact.png')});
  await page.locator('#node-ai-worker').click();
  await page.locator('#assistant-panel').waitFor();
  assert.match(await page.locator('#assistant-input').inputValue(), /visual-only worker/);
  assert.deepEqual(messages, []);
  await page.locator('#assistant-close').click();
  await openSettings();
  await page.locator('#settings-view').evaluate(view => {view.scrollTop = 0;});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({path: path.join(directory, 'settings-compact.png')});
  check('AI inspector tabs fit one row; compact submenu and Settings stay usable without generation');

  assert.equal((await storage()).includes(secret), false);
  assert.deepEqual(errors, []);
  assert.deepEqual(messages, []);
  check('No model generation, stored secret, page exception or browser console error');
  await writeFile(path.join(directory, 'settings-browser-results.json'), JSON.stringify({passed: checks.length, checks, errors, modelTraffic: false}, null, 2));
} catch (error) {
  await page.screenshot({path: path.join(directory, 'settings-failure.png')});
  throw error;
} finally {await browser.close();}
