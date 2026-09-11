const {chromium} = await import(process.env.WORKBENCH_PLAYWRIGHT_MODULE || 'playwright');
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';

const base = process.env.WORKBENCH_URL || 'http://127.0.0.1:8765/';
if (!['127.0.0.1', 'localhost'].includes(new URL(base).hostname)) throw new Error('Only local workbench verification is supported.');
const directory = process.env.WORKBENCH_BROWSER_ARTIFACTS || '/tmp/hyperspace-workbench-qa';
await mkdir(directory, {recursive: true});
const browser = await chromium.launch({channel: 'chrome', headless: true});
const page = await browser.newPage({viewport: {width: 1600, height: 1000}});
const errors = [];
const checks = [];
const sent = [];
const sentBodies = [];
const acceptedRequests = new Map();
let pending = false;
let cancelled = false;
let connected = true;
let failNextSubmission = false;
let expectedNetworkErrors = 0;
const proposal = {title: 'A visible review checkpoint', description: 'Move the coordinator and add a non-executable review stage.', operations: [
  {type: 'move_component', component_id: 'coordinator', x: 720, y: 160},
  {type: 'add_component', component_id: 'draft-review', name: 'Review checkpoint', description: 'A visual review stage only.', category: 'control', x: 960, y: 420},
  {type: 'connect', source: 'coordinator', target: 'draft-review', label: 'Review handoff', kind: 'control'},
  {type: 'set_note', component_id: 'coordinator', text: 'Review before continuing.'},
]};
let replyProposal = proposal;
let replyText = 'This is an offline UI fixture. <script>not executable</script>';
const check = name => { checks.push(name); console.log('PASS ' + name); };
const fulfill = (route, data, status = 200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(data)});
const savedWorkspace = () => page.evaluate(() => JSON.parse(localStorage.getItem('hyperspace-workbench-v1')));
const waitUntilIdle = () => page.waitForFunction(() => !document.querySelector('#assistant-send').disabled);
const highlight = async locator => {
  const selectedText = await locator.evaluate(node => {
    const range = document.createRange();
    range.selectNodeContents(node);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    return selection.toString();
  });
  assert.ok(selectedText.length > 0);
  await locator.click({button: 'right'});
  await page.locator('#assistant-panel').waitFor();
  assert.match(await page.locator('#assistant-context').innerText(), /Highlighted text/);
  assert.ok((await page.locator('#assistant-context-preview').textContent()).includes(selectedText));
  return selectedText;
};
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {
  if (message.type() !== 'error') return;
  if (expectedNetworkErrors > 0 && message.text() === 'Failed to load resource: net::ERR_FAILED' && message.location().url === new URL('/api/assistant/messages', base).href) {
    expectedNetworkErrors -= 1;
    return;
  }
  errors.push(message.text());
});
await page.route('**/api/assistant/**', route => {
  const url = new URL(route.request().url());
  if (url.pathname.endsWith('/status')) return fulfill(route, {authenticated: connected, available: connected, model: 'Offline test model', cli_version: 'test', csrf: 'offline-nonce'});
  if (url.pathname.endsWith('/messages')) {
    assert.equal(route.request().headers()['x-workbench-csrf'], 'offline-nonce');
    const body = route.request().postDataJSON();
    const serialized = route.request().postData();
    sent.push(body);
    sentBodies.push(serialized);
    if (acceptedRequests.has(body.request_id)) assert.equal(acceptedRequests.get(body.request_id), serialized);
    else acceptedRequests.set(body.request_id, serialized);
    cancelled = false;
    if (failNextSubmission) {
      failNextSubmission = false;
      return route.abort('failed');
    }
    return fulfill(route, {job_id: body.request_id, conversation_id: body.conversation_id || 'offline-conversation', status: 'running'}, 202);
  }
  if (url.pathname.endsWith('/cancel')) { cancelled = true; return fulfill(route, {status: 'cancelled'}); }
  return fulfill(route, {status: cancelled ? 'cancelled' : pending ? 'running' : 'completed', answer: replyText, proposal: cancelled || pending ? null : replyProposal, model: 'Offline test model'});
});

try {
  await page.goto(base);
  await page.waitForSelector('[data-node="coordinator"]');
  await page.locator('[data-node="coordinator"]').click({button: 'right'});
  await page.locator('#assistant-input').waitFor();
  await page.waitForFunction(() => !document.querySelector('#assistant-send').disabled);
  assert.match(await page.locator('#assistant-context').innerText(), /Coordinator/);
  assert.equal(sent.length, 0);
  check('Right-click opens subscription assistant with selected part, without generating');

  await page.locator('#assistant-close').click();
  await page.locator('[data-component="knowledge-store"]').click({modifiers: ['Shift']});
  assert.match(await page.locator('#selected-context').textContent(), /2 parts/);
  await page.locator('[data-node="coordinator"]').click({button: 'right'});
  assert.match(await page.locator('#assistant-context').innerText(), /Knowledge store/);
  await page.locator('#assistant-input').fill('Explain and improve these selected parts.');
  await page.locator('#assistant-send').click();
  await page.getByRole('button', {name: 'Preview changes', exact: true}).waitFor();
  assert.deepEqual(sent[0].selection.component_ids.sort(), ['coordinator', 'knowledge-store']);
  assert.equal(await page.locator('#assistant-messages script').count(), 0);
  assert.match(await page.locator('.chat-message.assistant .chat-text').innerText(), /<script>/);
  check('Multi-part context imports into safe, text-only chat replies');

  const initial = await page.locator('[data-node="coordinator"]').getAttribute('transform');
  const beforeWorkspace = await page.evaluate(() => localStorage.getItem('hyperspace-workbench-v1'));
  await page.getByRole('button', {name: 'Preview changes', exact: true}).click();
  assert.equal(await page.locator('#draft-preview-bar').isVisible(), true);
  assert.equal(await page.locator('#nodes-layer .node').count(), 13);
  assert.equal(await page.evaluate(() => localStorage.getItem('hyperspace-workbench-v1')), beforeWorkspace);
  await page.locator('#assistant-close').click();
  await page.screenshot({path: path.join(directory, 'assistant-draft-preview.png')});
  await page.locator('#draft-discard').click();
  assert.equal(await page.locator('#nodes-layer .node').count(), 12);
  assert.equal(await page.locator('[data-node="coordinator"]').getAttribute('transform'), initial);
  check('Proposal preview changes only a candidate, and dismiss restores the canvas');

  await page.locator('#assistant-open').click();
  await page.getByRole('button', {name: 'Preview changes', exact: true}).click();
  await page.screenshot({path: path.join(directory, 'assistant-chat.png')});
  await page.locator('#assistant-close').click();
  await page.locator('#draft-apply').click();
  assert.equal(await page.locator('#nodes-layer .node').count(), 13);
  assert.match(await page.locator('#draft-state-label').innerText(), /Visual draft/);
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('hyperspace-workbench-v1')));
  assert.equal(saved.notes.coordinator, 'Review before continuing.');
  await page.locator('#undo-button').click();
  assert.equal(await page.locator('#nodes-layer .node').count(), 12);
  assert.equal(await page.locator('[data-node="coordinator"]').getAttribute('transform'), initial);
  await page.locator('#redo-button').click();
  await page.reload();
  await page.waitForSelector('[data-node="draft-review"]');
  assert.equal(await page.locator('#nodes-layer .node').count(), 13);
  check('Apply is atomic, undo/redo restores the whole draft, and reload persists it');

  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu').waitFor();
  assert.equal(await page.locator('#node-menu').isVisible(), true);
  for (const name of ['Connect', 'Disconnect', 'Call AI', 'Delete', 'Duplicate']) assert.equal(await page.locator('#node-menu').getByRole('button', {name, exact: true}).isVisible(), true);
  await page.screenshot({path: path.join(directory, 'persona-node-menu.png')});
  await page.locator('#node-menu-duplicate').click();
  assert.equal(await page.locator('#nodes-layer .node').count(), 14);
  await page.locator('#draft-apply').click();
  await page.locator('#undo-button').click();
  assert.equal(await page.locator('#nodes-layer .node').count(), 13);
  check('Click opens five-action Persona menu; duplicate previews and undoes');

  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-delete').click();
  assert.equal(await page.locator('[data-node="coordinator"]').count(), 0);
  await page.locator('#draft-apply').click();
  await page.locator('#undo-button').click();
  assert.equal(await page.locator('[data-node="coordinator"]').count(), 1);
  check('Delete hides canonical parts in a reversible visual draft, not source');

  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-connect').click();
  await page.locator('#node-menu-target').selectOption('evidence-audit');
  await page.locator('#node-menu-edge-label').fill('Draft review flow');
  await page.locator('#node-menu-edge-kind').selectOption('feedback');
  await page.locator('#node-menu-detail').getByRole('button', {name: 'Preview change', exact: true}).click();
  await page.locator('#draft-apply').click();
  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-disconnect').click();
  const target = await page.locator('#node-menu-target option').filter({hasText: 'Coordinator → Independent audit'}).getAttribute('value');
  await page.locator('#node-menu-target').selectOption(target);
  await page.locator('#node-menu-detail').getByRole('button', {name: 'Preview change', exact: true}).click();
  await page.locator('#draft-apply').click();
  const disconnected = await page.evaluate(() => JSON.parse(localStorage.getItem('hyperspace-workbench-v1')));
  assert.equal(disconnected.drafts.mission.edges.some(edge => edge.source === 'coordinator' && edge.target === 'evidence-audit'), false);
  await page.locator('#undo-button').click();
  check('Connect and disconnect pick real endpoints and participate in undo');

  await page.locator('[data-node="coordinator"]').focus();
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('#node-menu').isVisible(), true);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#node-menu').isVisible(), false);
  await page.locator('[data-node="coordinator"]').click();
  await page.locator('#node-menu-ai').click();
  await page.locator('#node-ai-ask').click();
  assert.equal(await page.locator('#assistant-panel').isVisible(), true);
  pending = true;
  await page.waitForFunction(() => !document.querySelector('#assistant-send').disabled);
  await page.locator('#assistant-input').fill('Wait for this offline request.');
  await page.locator('#assistant-send').click();
  await page.locator('#assistant-stop').waitFor();
  await page.locator('#assistant-stop').click();
  await page.waitForFunction(() => document.querySelector('#assistant-messages').textContent.includes('Request stopped'));
  assert.equal(cancelled, true);
  await page.locator('#assistant-close').click();
  check('Keyboard menu, Call AI, and cancellation work without automatic replay');

  pending = false;
  cancelled = false;
  replyProposal = null;
  await page.locator('[data-component="coordinator"]').click();
  await page.locator('[data-inspector-tab="overview"]').click();
  const highlightedDescription = await highlight(page.locator('#inspector-content .component-description'));
  await waitUntilIdle();
  replyText = 'The highlighted description is attached to this offline reply.';
  await page.locator('#assistant-input').fill('Explain this highlighted description.');
  await page.locator('#assistant-send').click();
  await page.getByText(replyText, {exact: true}).waitFor();
  assert.equal(sent.at(-1).selection.selected_text, highlightedDescription);
  assert.deepEqual(sent.at(-1).selection.component_ids, ['coordinator']);
  await page.locator('#assistant-close').click();
  const sourceExcerpt = 'class Coordinator:\n    responsibility = "Coordinate visualized harness stages"\n';
  await page.route('**/api/components/coordinator/source', route => fulfill(route, {path: 'astra_harness/coordinator.py', content: sourceExcerpt, truncated: false}));
  await page.locator('[data-inspector-tab="source"]').click();
  const source = page.locator('#inspector-content .code-view');
  await source.waitFor();
  assert.equal(await source.textContent(), sourceExcerpt);
  const highlightedSource = await highlight(source);
  await waitUntilIdle();
  replyText = 'The highlighted source is attached to this offline reply.';
  await page.locator('#assistant-input').fill('Explain this highlighted source excerpt.');
  await page.locator('#assistant-send').click();
  await page.getByText(replyText, {exact: true}).waitFor();
  assert.equal(sent.at(-1).selection.selected_text, highlightedSource);
  assert.deepEqual(sent.at(-1).selection.component_ids, ['coordinator']);
  await page.locator('#assistant-close').click();
  check('Highlighted descriptions and synthetic source import exact text and selected component context');

  await page.locator('[data-component="draft-review"]').click();
  await page.locator('[data-inspector-tab="overview"]').click();
  await page.locator('.component-notes').fill('A draft-only note to preserve through reset undo.');
  await page.locator('[data-node="draft-review"]').focus();
  await page.keyboard.press('ArrowRight');
  const beforeReset = await savedWorkspace();
  assert.ok(beforeReset.positions.mission['draft-review']);
  assert.equal(beforeReset.notes['draft-review'], 'A draft-only note to preserve through reset undo.');
  await page.locator('#draft-reset').click();
  const reset = await savedWorkspace();
  assert.equal(reset.drafts.mission, undefined);
  assert.equal(reset.positions.mission?.['draft-review'], undefined);
  assert.equal(reset.notes['draft-review'], undefined);
  assert.equal(reset.notes.coordinator, beforeReset.notes.coordinator);
  assert.equal(await page.locator('#nodes-layer .node').count(), 12);
  await page.locator('#undo-button').click();
  assert.deepEqual(await savedWorkspace(), beforeReset);
  assert.equal(await page.locator('[data-node="draft-review"]').count(), 1);
  check('Reset removes draft-only positions and notes; undo restores the complete workspace');

  connected = false;
  const beforeUnavailable = sent.length;
  await page.locator('#assistant-open').click();
  await page.waitForFunction(() => document.querySelector('#assistant-status').textContent === 'Not connected');
  assert.equal(await page.locator('#assistant-send').isDisabled(), true);
  assert.match(await page.locator('#assistant-error').innerText(), /codex login/);
  await page.locator('#assistant-input').fill('This must not use an unavailable provider.');
  await page.locator('#assistant-input').press('Enter');
  assert.equal(sent.length, beforeUnavailable);
  await page.locator('#assistant-close').click();
  check('Unavailable ChatGPT status disables Send and keyboard submission without a POST');

  connected = true;
  await page.locator('#assistant-open').click();
  await waitUntilIdle();
  await page.locator('#assistant-new').click();
  const recoveryMessage = 'Recover this exact offline request only after I choose recovery.';
  replyText = 'The recovered request has exactly one offline answer.';
  const beforeRecovery = sent.length;
  const acceptedBeforeRecovery = acceptedRequests.size;
  expectedNetworkErrors = 1;
  failNextSubmission = true;
  const failedRequest = page.waitForEvent('requestfailed', request => request.method() === 'POST' && new URL(request.url()).pathname === '/api/assistant/messages');
  await page.locator('#assistant-input').fill(recoveryMessage);
  await page.locator('#assistant-send').click();
  await failedRequest;
  await page.locator('#assistant-recover').waitFor();
  assert.equal(sent.length, beforeRecovery + 1);
  const originalRequest = sent.at(-1);
  const originalBody = sentBodies.at(-1);
  assert.match(originalRequest.request_id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
  assert.equal(await page.locator('#assistant-send').isDisabled(), true);
  assert.equal(await page.locator('#assistant-new').isDisabled(), true);
  const storedPending = await page.evaluate(() => JSON.parse(sessionStorage.getItem('hyperspace-assistant-v1')).pending);
  assert.deepEqual(storedPending.body, originalRequest);
  await page.locator('#assistant-close').click();
  await page.reload();
  await page.waitForSelector('[data-node="coordinator"]');
  await page.locator('[data-component="knowledge-store"]').click();
  await page.locator('#assistant-open').click();
  await page.locator('#assistant-recover').waitFor();
  await page.waitForFunction(() => document.querySelector('#assistant-error').textContent.includes('nothing is retried automatically'));
  assert.match(await page.locator('#assistant-context').innerText(), /Knowledge store/);
  assert.equal(sent.length, beforeRecovery + 1);
  assert.equal(await page.locator('.chat-message.assistant').count(), 0);
  await page.locator('#assistant-recover').click();
  await page.getByText(replyText, {exact: true}).waitFor();
  await waitUntilIdle();
  assert.equal(sent.length, beforeRecovery + 2);
  assert.equal(sent.at(-1).request_id, originalRequest.request_id);
  assert.equal(sentBodies.at(-1), originalBody);
  assert.equal(acceptedRequests.size, acceptedBeforeRecovery + 1);
  assert.equal(await page.locator('.chat-message.assistant').count(), 1);
  assert.equal(await page.locator('.chat-message.user').count(), 1);
  assert.equal(await page.locator('#assistant-recover').isVisible(), false);
  const recovered = await page.evaluate(() => JSON.parse(sessionStorage.getItem('hyperspace-assistant-v1')));
  assert.equal(recovered.pending, null);
  assert.equal(recovered.job, null);
  await page.locator('#assistant-close').click();
  await page.locator('#assistant-open').click();
  await waitUntilIdle();
  assert.equal(sent.length, beforeRecovery + 2);
  assert.equal(await page.locator('.chat-message.assistant').count(), 1);
  await page.locator('#assistant-close').click();
  check('Ambiguous POST survives reload without replay; manual recovery reuses the exact body and yields one answer');

  await page.setViewportSize({width: 1024, height: 768});
  await page.locator('#fit-view').click();
  await page.locator('[data-node="coordinator"]').click();
  const menuBounds = await page.locator('#node-menu').boundingBox();
  assert.ok(menuBounds.x >= 0 && menuBounds.x + menuBounds.width <= 1024);
  assert.ok(menuBounds.y >= 0 && menuBounds.y + menuBounds.height <= 768);
  await page.screenshot({path: path.join(directory, 'persona-menu-compact.png')});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.deepEqual(errors, []);
  check('Compact menu stays on screen, with no browser errors or page overflow');
  await writeFile(path.join(directory, 'assistant-browser-results.json'), JSON.stringify({passed: checks.length, checks, errors, modelTraffic: false}, null, 2));
} finally { await browser.close(); }
