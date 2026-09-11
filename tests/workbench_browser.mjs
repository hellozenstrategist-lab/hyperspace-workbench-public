const { chromium } = await import(process.env.WORKBENCH_PLAYWRIGHT_MODULE || 'playwright');
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const artifactDirectory = process.env.WORKBENCH_BROWSER_ARTIFACTS || '/tmp/hyperspace-workbench-qa';
await mkdir(artifactDirectory, {recursive: true});
const artifactPath = (name) => path.join(artifactDirectory, name);
const workbenchUrl = process.env.WORKBENCH_URL || 'http://127.0.0.1:8765/';
if (!['127.0.0.1', 'localhost'].includes(new URL(workbenchUrl).hostname)) throw new Error('Browser verification requires a local workbench.');

const browser = await chromium.launch({channel: 'chrome', headless: true});
const page = await browser.newPage({viewport: {width: 1600, height: 1000}, deviceScaleFactor: 1});
const errors = [];
const checks = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
const check = (name) => { checks.push(name); console.log('PASS ' + name); };
try {
  await page.goto(workbenchUrl);
  await page.waitForFunction(() => document.querySelectorAll('.event-row').length > 0);
  assert.equal(await page.locator('#nodes-layer .node').count(), 12);
  assert.equal(await page.locator('#error-banner').isVisible(), false);
  check('Real mission snapshot and 12-component assembly load');
  await page.screenshot({path: artifactPath('assembly.png')});

  await page.locator('#component-search').fill('knowledge store');
  assert.equal(await page.locator('.component-item').count(), 1);
  await page.locator('.component-item').click();
  assert.equal(await page.locator('#inspector-title').textContent(), 'Knowledge store');
  await page.locator('#component-search').fill('');
  await page.locator('[data-inspector-tab="source"]').click();
  await page.locator('.source-listing').waitFor();
  assert.match(await page.locator('.source-listing .code-view').textContent(), /class KnowledgeStore/);
  await page.getByRole('button', {name: 'Expand source ↗'}).click();
  assert.equal(await page.locator('#artifact-dialog').isVisible(), true);
  assert.match(await page.locator('#artifact-content').textContent(), /class KnowledgeStore/);
  await page.locator('#artifact-close').click();
  check('Search, component inspection, source preview and expanded source');

  await page.locator('[data-inspector-tab="overview"]').click();
  const node = page.locator('[data-node="knowledge-store"]');
  const initialPosition = await node.getAttribute('transform');
  const bounds = await node.boundingBox();
  await page.mouse.move(bounds.x + 60, bounds.y + 20);
  await page.mouse.down();
  await page.mouse.move(bounds.x + 140, bounds.y + 55, {steps: 10});
  await page.mouse.up();
  assert.notEqual(await node.getAttribute('transform'), initialPosition);
  await page.locator('#undo-button').click();
  assert.equal(await node.getAttribute('transform'), initialPosition);
  await page.locator('#redo-button').click();
  assert.notEqual(await node.getAttribute('transform'), initialPosition);
  await page.locator('#reset-layout').click();
  check('Drag with grid snapping, undo, redo and reset');

  await page.locator('#explode-button').click();
  assert.equal(await page.locator('#explode-button').getAttribute('aria-pressed'), 'true');
  await page.locator('#isolate-button').click();
  assert.ok(await page.locator('#nodes-layer .node').count() < 12);
  await page.locator('#assembly-svg').focus();
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#nodes-layer .node').count(), 12);
  await page.locator('#explode-button').click();
  const zoom = await page.locator('#zoom-label').textContent();
  await page.locator('#zoom-in').click();
  assert.notEqual(await page.locator('#zoom-label').textContent(), zoom);
  await page.locator('#fit-view').click();
  check('Explode, isolate, escape, zoom and fit');

  await page.locator('#scene-select').selectOption('research');
  assert.equal(await page.locator('#nodes-layer .node').count(), 8);
  assert.equal(await page.locator('.node.event-active').count(), 0);
  await page.locator('.component-notes').fill('Inspect the head / QC boundary.');
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#save-workspace').click();
  const download = await downloadPromise;
  const file = artifactPath('exported-workspace.json');
  await download.saveAs(file);
  const workspace = JSON.parse(await readFile(file, 'utf8'));
  assert.equal(workspace.scene, 'research');
  assert.equal(workspace.notes['research-plan'], 'Inspect the head / QC boundary.');
  await page.locator('.component-notes').fill('Changed note');
  await page.locator('#scene-select').selectOption('mission');
  await page.locator('#workspace-file').setInputFiles(file);
  await page.waitForFunction(() => document.querySelector('#scene-select').value === 'research');
  assert.equal(await page.locator('.component-notes').inputValue(), 'Inspect the head / QC boundary.');
  await page.locator('#undo-button').click();
  assert.equal(await page.locator('#scene-select').inputValue(), 'mission');
  await page.locator('#scene-select').selectOption('research');
  assert.equal(await page.locator('.component-notes').inputValue(), 'Changed note');
  check('Independent research assembly, notes, export, import and full undo');

  await page.locator('#scene-select').selectOption('mission');
  await page.locator('#timeline-slider').evaluate(input => { input.value = '0'; input.dispatchEvent(new Event('input', {bubbles: true})); });
  assert.match(await page.locator('#timeline-position').textContent(), /^1 \/ /);
  await page.locator('#play-button').click();
  await page.waitForFunction(() => Number(document.querySelector('#timeline-slider').value) >= 2);
  await page.locator('#play-button').click();
  await page.locator('[data-inspector-tab="data"]').click();
  assert.ok(await page.locator('#inspector-content .code-view').count());
  await page.locator('#timeline-toggle').click();
  assert.match(await page.locator('#timeline-panel').getAttribute('class'), /collapsed/);
  await page.locator('#timeline-toggle').click();
  check('Recorded event scrubbing, playback, data inspection and drawer collapse');

  await page.locator('[data-view="dashboard"]').click();
  assert.equal(await page.locator('#dashboard-view').isVisible(), true);
  assert.equal(await page.locator('#assembly-view').isVisible(), false);
  assert.ok(await page.locator('#dashboard-content .metric-card').count() >= 8);
  await page.locator('.run-search').fill('example-run');
  assert.equal(await page.locator('.run-name').count(), 1);
  await page.locator('.run-search').fill('');
  await page.locator('[aria-label="Compare selected run with"]').selectOption({index: 1});
  assert.ok(await page.locator('.comparison-body table').count());
  await page.locator('[aria-label="Filter artifacts"]').fill('report.json');
  await page.locator('.artifact-button').first().click();
  await page.waitForFunction(() => !document.querySelector('#artifact-content').textContent.includes('Reading saved artifact'));
  assert.ok((await page.locator('#artifact-content').textContent()).length > 30);
  await page.locator('#artifact-close').click();
  await page.locator('#dashboard-view').evaluate(element => element.scrollTop = 0);
  await page.screenshot({path: artifactPath('dashboard.png')});
  check('Live inventory, filters, recorded-run comparison and saved artifact preview');

  await page.locator('[data-view="geometry"]').click();
  assert.ok(await page.locator('.geometry-point').count() > 0);
  assert.ok(await page.locator('.neighbor-row').count() > 0);
  const initialClaim = await page.locator('.geometry-claim').textContent();
  await page.locator('.neighbor-row').first().click();
  assert.notEqual(await page.locator('.geometry-claim').textContent(), initialClaim);
  await page.screenshot({path: artifactPath('geometry.png')});
  check('Saved Poincaré coordinates and selectable nearest-neighbor inspection');

  await page.locator('[data-view="assembly"]').click();
  await page.locator('#inspector-close').click();
  assert.equal(await page.locator('.inspector-panel').isVisible(), false);
  await page.locator('.component-item').filter({hasText: 'Coordinator'}).click();
  assert.equal(await page.locator('.inspector-panel').isVisible(), true);
  await page.setViewportSize({width: 1024, height: 768});
  await page.locator('#fit-view').click();
  assert.equal(await page.locator('#nodes-layer .node').count(), 12);
  await page.screenshot({path: artifactPath('compact.png')});
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  assert.equal(overflow, false);
  check('Inspector collapse/reopen and 1024px layout without horizontal overflow');

  const researchId = await page.locator('#run-select optgroup[label="Research snapshots"] option').first().getAttribute('value');
  const researchFixture = {
    id: researchId, name: 'Synthetic research UI fixture', kind: 'research',
    relative_path: 'test-fixture', status: 'paused', updated_at: '2025-01-01T12:00:00Z',
    provider: 'openrouter', model: 'fixture/head', rounds: 1, requests: 4, tokens: 1234,
    config: {workers: 3, models: {head: 'fixture/head', worker: 'fixture/worker', qc: 'fixture/qc'}},
    metrics: {deliveries: 0, usage: {usage_complete: false}},
    workers: ['head', 'worker/agent-a', 'worker/agent-b', 'worker/agent-c', 'qc'].map(id => ({id, state: 'completed', model: 'fixture/model'})),
    events: ['plan', 'work', 'synthesize', 'review'].map((phase, index) => ({id: `rounds/001/${phase}/attempt-001/runtime_state.json:agent-a:0`, kind: 'request_completed', at: `2025-01-01T12:00:0${index}Z`, agent: 'agent-a', summary: `${phase} · fixture/model`, data: {phase}})),
    phases: ['plan', 'work', 'synthesize', 'review'].map(name => ({name, status: 'completed', round: 1, model: 'fixture/model'})),
    deliveries: [], nodes: [], edges: [], artifacts: [], warnings: [],
  };
  await page.route(`**/api/runs/${researchId}`, route => route.fulfill({json: researchFixture}));
  await page.locator('#run-select').selectOption(researchId);
  await page.waitForFunction(() => document.querySelector('#scene-select').value === 'research');
  assert.equal(await page.locator('#nodes-layer .node').count(), 8);
  assert.equal(await page.locator('.node.event-active').getAttribute('data-node'), 'research-qc');
  await page.locator('#timeline-slider').evaluate(input => { input.value = '2'; input.dispatchEvent(new Event('input', {bubbles: true})); });
  assert.equal(await page.locator('.node.event-active').getAttribute('data-node'), 'research-synthesis');
  await page.locator('[data-view="dashboard"]').click();
  assert.match(await page.locator('#selected-run-detail').textContent(), /Usage is incomplete/);
  assert.match(await page.locator('#selected-run-detail').textContent(), /5/);
  await page.locator('[data-view="geometry"]').click();
  assert.match(await page.locator('#geometry-content').textContent(), /artifacts, not geometric routing/);
  check('Synthetic research role states, synthesis mapping, incomplete usage and separate storage');

  assert.deepEqual(errors, []);
  check('No browser console errors or page exceptions');
  await writeFile(artifactPath('verification.json'), JSON.stringify({passed: true, checks, console_errors: errors, generated_at: new Date().toISOString()}, null, 2));
} catch (error) {
  await page.screenshot({path: artifactPath('failure.png')});
  console.error(error);
  console.error(JSON.stringify({checks, errors}));
  process.exitCode = 1;
} finally { await browser.close(); }
