const select = (selector) => document.querySelector(selector);
const selectAll = (selector) => [...document.querySelectorAll(selector)];
const SVG_NS = 'http://www.w3.org/2000/svg';
const STORAGE_KEY = 'hyperspace-workbench-v1';
const NODE_WIDTH = 180;
const NODE_HEIGHT = 82;
const PREFERENCES_KEY = 'hyperspace-preferences-v1';
const DEFAULT_PREFERENCES = {snap_to_grid: true, show_guides: true, decorative_art: true, reduced_motion: false, auto_refresh: false, refresh_interval: 15};
let preferences = {...DEFAULT_PREFERENCES};
let refreshTimer;
const COLORS = {control: '#0055ff', runtime: '#d42adc', storage: '#d42adc', routing: '#0055ff', interface: '#0055ff', research: '#4b6800'};
const SYMBOLS = {control: 'CT', runtime: 'RT', storage: 'DB', routing: 'RX', interface: 'IO', research: 'AI'};
const state = {
  overview: null, run: null, runId: null, sceneId: 'mission', selected: 'coordinator',
  view: 'assembly', tab: 'overview', isolated: false, exploded: false, positions: {},
  camera: {x: 0, y: 0, scale: 1}, history: [], future: [], eventIndex: -1,
  playing: null, refreshing: false, runSequence: 0, inspectorSequence: 0, artifactSequence: 0, pendingRunId: null,
  geometrySelection: null, comparisonId: null, notes: {}, saved: false,
  selection: new Set(['coordinator']), drafts: {}, preview: null, aiParts: {},
};

function changePreferences(patch, save = true) {
  const next = {...preferences};
  for (const [key, value] of Object.entries(patch)) {
    if (!Object.hasOwn(DEFAULT_PREFERENCES, key)) continue;
    if (key === 'refresh_interval' ? [5, 15, 30, 60].includes(value) : typeof value === 'boolean') next[key] = value;
  }
  if (save) {
    try { localStorage.setItem(PREFERENCES_KEY, JSON.stringify(next)); }
    catch { toast('Preferences could not be saved in this browser.'); return {...preferences}; }
  }
  preferences = next;
  document.body.classList.toggle('prefs-no-art', !next.decorative_art);
  document.body.classList.toggle('prefs-reduced-motion', next.reduced_motion);
  select('.canvas-guidebar').hidden = !next.show_guides;
  select('#live-toggle').checked = next.auto_refresh;
  clearInterval(refreshTimer);
  refreshTimer = setInterval(() => {
    if (preferences.auto_refresh && !document.hidden && !state.playing && !state.preview && state.view !== 'settings' && !document.activeElement?.matches('textarea, select, input:not([type="checkbox"])')) refresh();
  }, next.refresh_interval * 1000);
  window.dispatchEvent(new CustomEvent('workbench:preferences-changed', {detail: {...next}}));
  return {...next};
}

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined && text !== null) item.textContent = String(text);
  return item;
}

function svgElement(tag, attributes = {}, text) {
  const item = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, String(value));
  if (text !== undefined) item.textContent = text;
  return item;
}

function action(label, callback, className = 'secondary-button') {
  const button = element('button', className, label);
  button.type = 'button';
  button.addEventListener('click', callback);
  return button;
}

function short(value, limit = 25) {
  const text = String(value ?? '');
  return text.length > limit ? text.slice(0, limit - 1) + '…' : text;
}

function human(value) { return String(value ?? 'unknown').replaceAll('_', ' ').replaceAll('-', ' '); }
function number(value) { return new Intl.NumberFormat().format(Number.isFinite(Number(value)) ? Number(value) : 0); }
function compact(value) { return new Intl.NumberFormat(undefined, {notation: 'compact', maximumFractionDigits: 1}).format(Number(value) || 0); }
function timestamp(value, timeOnly = false) {
  if (value === '' || value === null || value === undefined) return 'Time unavailable';
  const date = new Date(typeof value === 'number' && value < 100000000000 ? value * 1000 : value);
  if (Number.isNaN(date.getTime())) return 'Time unavailable';
  return timeOnly ? date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'}) : date.toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
}

function statusClass(value) {
  if (/completed|accepted|passed|incorporated|acknowledged|success/.test(value)) return 'success';
  if (/failed|error|blocked|uncertain/.test(value)) return 'warning';
  if (/running|active|attempting/.test(value)) return 'active';
  return 'neutral';
}

function badge(value) { return element('span', 'tag ' + statusClass(String(value)), human(value)); }
function empty(message, detail = '') {
  const box = element('div', 'empty-state');
  box.append(element('strong', '', message));
  if (detail) box.append(element('p', '', detail));
  return box;
}

function section(title) {
  const box = element('section', 'inspector-section');
  box.append(element('h3', 'section-title', title));
  return box;
}

function keyValue(label, value) {
  const row = element('div', 'kv-row');
  row.append(element('span', '', label), element('span', 'kv-value', value));
  return row;
}

function codeBlock(value) {
  const block = element('pre', 'code-view', typeof value === 'string' ? value : JSON.stringify(value, null, 2));
  block.tabIndex = 0;
  return block;
}

async function api(path) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, {signal: controller.signal, cache: 'no-store', credentials: 'same-origin'});
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Request failed (${response.status})`);
    }
    return await response.json();
  } finally { clearTimeout(timer); }
}

let toastTimer;
function toast(message) {
  const box = select('#toast');
  box.textContent = message;
  box.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove('visible'), 3400);
}

function showError(error) {
  select('#error-banner').hidden = false;
  select('#error-banner').textContent = error.name === 'AbortError' ? 'The local reader took too long. Use Refresh to try again.' : error.message;
  select('#connection-status').textContent = 'Reader unavailable';
}

function catalog() { return state.overview?.catalog; }
function baseScene(identifier = state.sceneId) { return catalog()?.scenes.find((item) => item.id === identifier); }
function effectiveWorkspace() { return state.preview?.candidate || workspace(); }
function scene() {
  const base = baseScene();
  if (!base) return null;
  const draft = effectiveWorkspace().drafts?.[state.sceneId];
  return draft ? {...base, nodes: [...base.nodes.filter(node => !draft.hidden?.includes(node.id)), ...(draft.nodes || [])], edges: draft.edges.filter(edge => !draft.hidden?.includes(edge.source) && !draft.hidden?.includes(edge.target))} : base;
}
function reconcileSelection() {
  const identifiers = new Set(scene()?.nodes.map(node => node.id) || []);
  if (!identifiers.has(state.selected)) state.selected = [...identifiers][0] || null;
  state.selection = new Set([...state.selection].filter(identifier => identifiers.has(identifier)));
  if (!state.selection.size && state.selected) state.selection.add(state.selected);
}
function component(identifier) {
  const original = catalog()?.components.find((item) => item.id === identifier);
  if (original) return original;
  for (const draft of Object.values(effectiveWorkspace().drafts || {})) {
    const node = draft.nodes?.find((item) => item.id === identifier);
    if (node) return {...node, subtitle: 'Visual draft stage', source: null, inputs: [], outputs: [], invariants: ['This stage exists in the visual draft. The executable harness is unchanged.']};
  }
  return null;
}
function currentEvent() { return state.run?.events?.[state.eventIndex]; }

function loadPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (saved?.version === 1) {
      validateWorkspace(saved);
      state.positions = saved.positions || {};
      state.notes = saved.notes || {};
      state.drafts = saved.drafts || {};
      state.aiParts = saved.ai_parts || {};
      state.sceneId = saved.scene === 'research' ? 'research' : 'mission';
      state.saved = true;
    }
  } catch { state.positions = {}; state.notes = {}; state.drafts = {}; state.aiParts = {}; }
}

function workspace() {
  return {version: 1, name: 'Hyperspace Workbench', scene: state.sceneId, positions: state.positions, notes: state.notes, drafts: state.drafts, ...(Object.keys(state.aiParts).length ? {ai_parts: state.aiParts} : {})};
}

function validFields(value, allowed) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).every(key => allowed.includes(key));
}

function validateWorkspace(value) {
  if (!validFields(value, ['version', 'name', 'scene', 'positions', 'notes', 'drafts', 'ai_parts']) || value.version !== 1 || !value.positions || typeof value.positions !== 'object' || Array.isArray(value.positions)) throw new Error('Choose a Hyperspace workspace JSON file.');
  if (Object.hasOwn(value, 'name') && (typeof value.name !== 'string' || value.name.length > 100)) throw new Error('Invalid workspace name.');
  if (Object.hasOwn(value, 'scene') && !['mission', 'research'].includes(value.scene)) throw new Error('Invalid workspace scene.');
  validateDrafts(value.drafts ?? {});
  if (Object.hasOwn(value, 'ai_parts')) {
    if (!value.ai_parts || typeof value.ai_parts !== 'object' || Array.isArray(value.ai_parts) || Object.keys(value.ai_parts).length > 100) throw new Error('Invalid part AI profiles.');
    for (const [identifier, profile] of Object.entries(value.ai_parts)) {
      if (!/^[a-z][a-z0-9-]{0,60}$/.test(identifier) || !validFields(profile, ['enabled', 'primary_model', 'fallback_model', 'max_steps', 'temperature']) || typeof profile.enabled !== 'boolean' || typeof profile.primary_model !== 'string' || profile.primary_model.length > 200 || typeof profile.fallback_model !== 'string' || profile.fallback_model.length > 200 || !Number.isInteger(profile.max_steps) || profile.max_steps < 1 || profile.max_steps > 100 || !Number.isFinite(profile.temperature) || profile.temperature < 0 || profile.temperature > 2) throw new Error('Invalid visual AI profile.');
    }
  }
  for (const [sceneId, positions] of Object.entries(value.positions)) {
    if (!['mission', 'research'].includes(sceneId) || !positions || typeof positions !== 'object' || Array.isArray(positions)) throw new Error('Invalid workspace scene.');
    if (Object.keys(positions).length > 100) throw new Error('Too many workspace components.');
    const known = new Set([...(baseScene(sceneId)?.nodes || []), ...(value.drafts?.[sceneId]?.nodes || [])].map(node => node.id));
    for (const [identifier, point] of Object.entries(positions)) {
      if (!/^[a-z][a-z0-9-]{0,60}$/.test(identifier) || !validFields(point, ['x', 'y']) || !Number.isFinite(point.x) || !Number.isFinite(point.y) || Math.abs(point.x) > 20000 || Math.abs(point.y) > 20000 || (catalog() && !known.has(identifier))) throw new Error('Invalid component position.');
    }
  }
  if (Object.hasOwn(value, 'notes') && (!value.notes || typeof value.notes !== 'object' || Array.isArray(value.notes) || Object.keys(value.notes).length > 100 || Object.entries(value.notes).some(([key, note]) => !/^[a-z][a-z0-9-]{0,60}$/.test(key) || typeof note !== 'string' || note.length > 4000))) throw new Error('Invalid component notes.');
  if (catalog()) {
    const known = new Set([...catalog().components, ...Object.values(value.drafts || {}).flatMap(draft => draft.nodes)].map(part => part.id));
    if (Object.keys(value.notes || {}).some(identifier => !known.has(identifier))) throw new Error('A note refers to an unknown component.');
    if (Object.keys(value.ai_parts || {}).some(identifier => !known.has(identifier))) throw new Error('An AI profile refers to an unknown component.');
  }
}

function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(workspace())); state.saved = true; }
  catch { toast('Browser storage is unavailable. Export your workspace to keep it.'); }
}

function download(name, value) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'}));
  const link = element('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function pointFor(identifier) {
  const original = scene()?.nodes.find((node) => node.id === identifier) || {x: 0, y: 0};
  const saved = effectiveWorkspace().positions[state.sceneId]?.[identifier] || original;
  return {x: saved.x * (state.exploded ? 1.45 : 1), y: saved.y * (state.exploded ? 1.45 : 1)};
}

function setPoint(identifier, point) {
  state.positions[state.sceneId] ||= {};
  const factor = state.exploded ? 1.45 : 1;
  const grid = preferences.snap_to_grid ? 8 : 1;
  state.positions[state.sceneId][identifier] = {x: Math.round(point.x / factor / grid) * grid, y: Math.round(point.y / factor / grid) * grid};
}

function snapshotLayout() { return JSON.stringify({positions: state.positions, exploded: state.exploded, notes: state.notes, scene: state.sceneId, drafts: state.drafts, ai_parts: state.aiParts}); }
function remember(before) {
  if (before === snapshotLayout()) return;
  state.history.push(before);
  if (state.history.length > 60) state.history.shift();
  state.future = [];
  persist();
  updateToolbar();
}

function undo(redo = false) {
  dismissPreview();
  const from = redo ? state.future : state.history;
  const to = redo ? state.history : state.future;
  if (!from.length) return;
  to.push(snapshotLayout());
  const saved = JSON.parse(from.pop());
  state.positions = saved.positions;
  state.exploded = saved.exploded;
  state.notes = saved.notes;
  state.drafts = saved.drafts || {};
  state.aiParts = saved.ai_parts || {};
  if (state.sceneId !== saved.scene) {
    state.sceneId = saved.scene;
    state.selected = saved.scene === 'research' ? 'research-plan' : 'coordinator';
    state.isolated = false;
    select('#scene-select').value = saved.scene;
  }
  reconcileSelection();
  persist();
  renderExplorer();
  renderDiagram();
  renderInspector();
  updateToolbar();
  requestAnimationFrame(fitView);
}

function neighbors(identifier) {
  const connected = new Set([identifier]);
  for (const edge of scene()?.edges || []) {
    if (edge.source === identifier) connected.add(edge.target);
    if (edge.target === identifier) connected.add(edge.source);
  }
  return connected;
}

function visibleNodes() {
  const connected = state.isolated ? neighbors(state.selected) : null;
  return (scene()?.nodes || []).filter((node) => !connected || connected.has(node.id));
}

function activeComponent(event = currentEvent()) {
  if (!event || state.run?.kind !== state.sceneId) return null;
  if (state.sceneId === 'research') {
    const context = `${event.summary || ''} ${event.id || ''}`;
    if (/synthesize|synthesis/.test(context)) return 'research-synthesis';
    if (/review|qc/.test(context)) return 'research-qc';
    if (/plan/.test(context)) return 'research-plan';
    if (/work|collect|falsif/.test(context)) return 'research-work';
    return 'openrouter-runtime';
  }
  const kind = event.kind || '';
  if (/rout/.test(kind)) return 'router';
  if (/deliver|ack|inbox|steer|message|runtime_accepted/.test(kind)) return 'event-bus';
  if (/publish|knowledge|coordinate/.test(kind)) return 'knowledge-store';
  if (/index|neighbor|projection|hyperbolic_query/.test(kind)) return 'hyperspace-index';
  if (/notifi|photon/.test(kind)) return 'notifications';
  if (/assign|attention|novel/.test(kind)) return 'attention';
  if (/audit|verified|incorporat/.test(kind)) return 'evidence-audit';
  if (/turn|worker|tool|usage|request/.test(kind)) return state.run?.provider === 'openrouter' ? 'openrouter-runtime' : 'codex-runtime';
  return 'coordinator';
}

function edgePath(source, target) {
  const sourcePoint = pointFor(source);
  const targetPoint = pointFor(target);
  const horizontal = Math.abs(targetPoint.x - sourcePoint.x) > Math.abs(targetPoint.y - sourcePoint.y) * 0.85;
  if (horizontal) {
    const direction = targetPoint.x >= sourcePoint.x ? 1 : -1;
    const start = {x: sourcePoint.x + (direction > 0 ? NODE_WIDTH : 0), y: sourcePoint.y + NODE_HEIGHT / 2};
    const end = {x: targetPoint.x + (direction > 0 ? 0 : NODE_WIDTH), y: targetPoint.y + NODE_HEIGHT / 2};
    const bend = Math.max(45, Math.abs(end.x - start.x) * 0.5);
    return `M${start.x},${start.y} C${start.x + bend * direction},${start.y} ${end.x - bend * direction},${end.y} ${end.x},${end.y}`;
  }
  const direction = targetPoint.y >= sourcePoint.y ? 1 : -1;
  const start = {x: sourcePoint.x + NODE_WIDTH / 2, y: sourcePoint.y + (direction > 0 ? NODE_HEIGHT : 0)};
  const end = {x: targetPoint.x + NODE_WIDTH / 2, y: targetPoint.y + (direction > 0 ? 0 : NODE_HEIGHT)};
  const bend = Math.max(40, Math.abs(end.y - start.y) * 0.5);
  return `M${start.x},${start.y} C${start.x},${start.y + bend * direction} ${end.x},${end.y - bend * direction} ${end.x},${end.y}`;
}

function renderDiagram() {
  if (!scene()) return;
  const nodes = visibleNodes();
  const visible = new Set(nodes.map((node) => node.id));
  const active = activeComponent();
  const search = select('#component-search').value.toLowerCase();
  const edgesLayer = select('#edges-layer');
  edgesLayer.replaceChildren();
  for (const edge of scene().edges) {
    if (!visible.has(edge.source) || !visible.has(edge.target)) continue;
    const originalEdges = state.drafts[state.sceneId]?.edges || baseScene().edges;
    const added = state.preview && !originalEdges.some((item) => item.source === edge.source && item.target === edge.target && item.label === edge.label);
    const group = svgElement('g', {class: 'edge ' + edge.kind + (active && [edge.source, edge.target].includes(active) ? ' active' : '') + (added ? ' preview-added' : '')});
    group.append(svgElement('title', {}, `${component(edge.source)?.name} → ${component(edge.target)?.name}: ${edge.label}`));
    group.append(svgElement('path', {d: edgePath(edge.source, edge.target), 'marker-end': `url(#arrow-${edge.kind})`, fill: 'none'}));
    edgesLayer.append(group);
  }
  const nodesLayer = select('#nodes-layer');
  nodesLayer.replaceChildren();
  for (const node of nodes) {
    const part = component(node.id);
    const point = pointFor(node.id);
    const isSelected = state.selection.has(node.id);
    const dimmed = search && !`${part.name} ${part.subtitle} ${part.category}`.toLowerCase().includes(search);
    const changed = state.preview?.operations.some((operation) => operation.component_id === node.id);
    const group = svgElement('g', {class: `node ${part.category}${isSelected ? ' selected multi-selected' : ''}${dimmed ? ' dimmed' : ''}${node.id === active ? ' event-active' : ''}${node.id.startsWith('draft-') ? ' draft-node' : ''}${changed ? ' preview-added' : ''}`, transform: `translate(${point.x},${point.y})`, 'data-node': node.id, role: 'button', tabindex: '0', 'aria-label': `${part.name}: ${part.subtitle}`, 'aria-pressed': isSelected});
    group.append(svgElement('title', {}, part.description));
    group.append(svgElement('rect', {width: NODE_WIDTH, height: NODE_HEIGHT, rx: 8}));
    group.append(svgElement('rect', {x: 12, y: 13, width: 25, height: 25, rx: 5, fill: COLORS[part.category], opacity: 0.13, stroke: 'none', class: 'node-icon-bg'}));
    group.append(svgElement('text', {x: 24.5, y: 30, fill: COLORS[part.category], 'text-anchor': 'middle', class: 'node-symbol'}, SYMBOLS[part.category]));
    group.append(svgElement('text', {x: 45, y: 29, class: 'node-title'}, short(part.name, 19)));
    group.append(svgElement('text', {x: 13, y: 53, class: 'node-subtitle'}, short(part.subtitle, 28)));
    group.append(svgElement('circle', {cx: 16, cy: 69, r: 2.5, fill: node.id === active ? '#59d9ce' : COLORS[part.category]}));
    group.append(svgElement('text', {x: 24, y: 72, class: 'node-status'}, node.id.startsWith('draft-') ? 'VISUAL DRAFT' : node.id === active ? 'RECORDED EVENT' : part.category.toUpperCase()));
    for (const port of [[0, NODE_HEIGHT / 2], [NODE_WIDTH, NODE_HEIGHT / 2]]) group.append(svgElement('circle', {cx: port[0], cy: port[1], r: 3.2, class: 'node-port'}));
    nodesLayer.append(group);
  }
  select('#canvas-caption').textContent = state.isolated ? `${component(state.selected)?.name} / connected components` : scene().description;
  select('#selection-hint').textContent = state.isolated ? 'Focused parts · Escape to return' : 'Right-click a part to ask AI · Shift-click to select several';
  updateCamera();
  renderMinimap();
}

function updateCamera() {
  const {x, y, scale} = state.camera;
  select('#viewport').setAttribute('transform', `translate(${x},${y}) scale(${scale})`);
  select('#zoom-label').textContent = Math.round(scale * 100) + '%';
  select('#canvas-wrap').style.backgroundSize = `${24 * scale}px ${24 * scale}px`;
  select('#canvas-wrap').style.backgroundPosition = `${x}px ${y}px`;
  renderMinimap();
}

function fitView() {
  if (state.view !== 'assembly') return;
  const bounds = select('#assembly-svg').getBoundingClientRect();
  const nodes = visibleNodes();
  if (!bounds.width || !bounds.height || !nodes.length) return;
  const points = nodes.map((node) => pointFor(node.id));
  const minX = Math.min(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const width = Math.max(...points.map((point) => point.x)) - minX + NODE_WIDTH;
  const height = Math.max(...points.map((point) => point.y)) - minY + NODE_HEIGHT;
  const scale = Math.min(1.2, Math.max(0.1, Math.min((bounds.width - 90) / width, (bounds.height - 100) / height)));
  state.camera = {scale, x: (bounds.width - width * scale) / 2 - minX * scale, y: (bounds.height - height * scale) / 2 - minY * scale};
  updateCamera();
}

function zoom(factor, center) {
  const bounds = select('#assembly-svg').getBoundingClientRect();
  const anchor = center || {x: bounds.width / 2, y: bounds.height / 2};
  const next = Math.min(2.8, Math.max(0.12, state.camera.scale * factor));
  const ratio = next / state.camera.scale;
  state.camera.x = anchor.x - (anchor.x - state.camera.x) * ratio;
  state.camera.y = anchor.y - (anchor.y - state.camera.y) * ratio;
  state.camera.scale = next;
  updateCamera();
}

function renderMinimap() {
  const container = select('#minimap');
  if (!container || !scene()) return;
  const nodes = visibleNodes();
  const points = nodes.map((node) => pointFor(node.id));
  if (!points.length) return;
  const minX = Math.min(...points.map((point) => point.x)) - 70;
  const minY = Math.min(...points.map((point) => point.y)) - 70;
  const width = Math.max(...points.map((point) => point.x)) + NODE_WIDTH + 70 - minX;
  const height = Math.max(...points.map((point) => point.y)) + NODE_HEIGHT + 70 - minY;
  const map = svgElement('svg', {viewBox: `${minX} ${minY} ${width} ${height}`, width: '100%', height: '100%', 'aria-label': 'Assembly overview'});
  nodes.forEach((node, index) => map.append(svgElement('rect', {x: points[index].x, y: points[index].y, width: NODE_WIDTH, height: NODE_HEIGHT, rx: 8, fill: node.id === state.selected ? '#59d9ce' : '#394454'})));
  const bounds = select('#assembly-svg').getBoundingClientRect();
  map.append(svgElement('rect', {x: -state.camera.x / state.camera.scale, y: -state.camera.y / state.camera.scale, width: bounds.width / state.camera.scale, height: bounds.height / state.camera.scale, fill: '#59d9ce', 'fill-opacity': 0.04, stroke: '#59d9ce', 'stroke-width': 5}));
  container.replaceChildren(map);
}

function renderExplorer() {
  if (!scene()) return;
  const list = select('#component-list');
  const query = select('#component-search').value.toLowerCase();
  list.replaceChildren();
  const groups = new Map();
  for (const node of scene().nodes) {
    const part = component(node.id);
    if (!`${part.name} ${part.subtitle} ${part.category}`.toLowerCase().includes(query)) continue;
    if (!groups.has(part.category)) groups.set(part.category, []);
    groups.get(part.category).push(part);
  }
  for (const [category, parts] of groups) {
    list.append(element('div', 'explorer-group-label', `${category.toUpperCase()} / ${parts.length}`));
    for (const part of parts) {
      const button = action('', (event) => selectComponent(part.id, {additive: event.shiftKey}), 'component-item' + (state.selection.has(part.id) ? ' selected' : ''));
      button.dataset.component = part.id;
      button.setAttribute('aria-pressed', state.selection.has(part.id));
      const dot = element('span', 'component-dot ' + category);
      dot.style.background = COLORS[category];
      const copy = element('span', 'component-copy');
      copy.append(element('span', 'component-name', part.name), element('span', 'component-meta', part.source?.split('/').pop() || 'Visual draft stage'));
      button.append(dot, copy);
      list.append(button);
    }
  }
  if (!groups.size) list.append(empty('No matching components', 'Try a module name or category.'));
  select('#explorer-stats').replaceChildren(keyValue('Components', scene().nodes.length), keyValue('Connections', scene().edges.length), keyValue('Saved runs', state.overview.counts.runs));
}

function selectComponent(identifier, {additive = false} = {}) {
  if (!component(identifier)) return;
  if (additive && !state.selection.has(identifier) && state.selection.size >= 8) { toast('Select up to eight parts for one conversation.'); return; }
  if (additive && state.selection.has(identifier) && state.selection.size > 1) {
    state.selection.delete(identifier);
    identifier = [...state.selection].at(-1);
  } else if (additive) state.selection.add(identifier);
  else state.selection = new Set([identifier]);
  state.selected = identifier;
  if (!scene().nodes.some((node) => node.id === identifier)) {
    state.sceneId = catalog().scenes.find((item) => item.nodes.some((node) => node.id === identifier))?.id || 'mission';
    select('#scene-select').value = state.sceneId;
    state.isolated = false;
  }
  document.body.classList.add('inspector-open');
  document.body.classList.remove('inspector-collapsed');
  renderExplorer();
  renderDiagram();
  renderInspector();
  updateToolbar();
}

function updateToolbar() {
  select('#explode-button').classList.toggle('active', state.exploded);
  select('#explode-button').setAttribute('aria-pressed', state.exploded);
  select('#isolate-button').classList.toggle('active', state.isolated);
  select('#isolate-button').setAttribute('aria-pressed', state.isolated);
  select('#isolate-button').disabled = !state.selected;
  select('#undo-button').disabled = !state.history.length;
  select('#redo-button').disabled = !state.future.length;
  select('#reset-layout').disabled = Boolean(state.preview);
  select('#explode-button').disabled = Boolean(state.preview);
  if (select('#selected-context')) select('#selected-context').textContent = `${state.selection.size} ${state.selection.size === 1 ? 'part' : 'parts'} selected`;
  if (select('#draft-state-label')) select('#draft-state-label').textContent = state.preview ? 'Preview · not applied' : state.drafts[state.sceneId] ? 'Visual draft' : 'Saved architecture';
  if (select('#draft-reset')) select('#draft-reset').hidden = !state.drafts[state.sceneId] || Boolean(state.preview);
  window.dispatchEvent(new CustomEvent('workbench:selection'));
}

function switchScene(identifier) {
  dismissPreview(false);
  state.sceneId = identifier;
  state.isolated = false;
  state.selected = identifier === 'research' ? 'research-plan' : 'coordinator';
  state.selection = new Set([state.selected]);
  reconcileSelection();
  state.tab = 'overview';
  select('#scene-select').value = identifier;
  renderExplorer(); renderDiagram(); renderInspector(); updateToolbar();
  persist();
  requestAnimationFrame(fitView);
}

function switchView(view) {
  if (!['assembly', 'dashboard', 'geometry', 'settings'].includes(view)) return;
  state.view = view;
  closeNodeMenu();
  for (const name of ['assembly', 'dashboard', 'geometry', 'settings']) if (select(`#${name}-view`)) select(`#${name}-view`).hidden = name !== view;
  document.body.classList.toggle('settings-open', view === 'settings');
  select('#timeline-panel').hidden = view === 'settings';
  for (const button of selectAll('[data-view]')) {
    button.classList.toggle('active', button.dataset.view === view);
    button.setAttribute('aria-pressed', button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  }
  select('#view-title').textContent = {assembly: 'Assembly', dashboard: 'Run dashboard', geometry: 'Knowledge geometry', settings: 'Settings'}[view];
  if (view === 'assembly') requestAnimationFrame(fitView);
  if (view === 'dashboard') renderDashboard();
  if (view === 'geometry') renderGeometry();
  window.dispatchEvent(new CustomEvent('workbench:view', {detail: {view}}));
}

function initializeCanvas() {
  const canvas = select('#assembly-svg');
  let gesture = null;
  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || !scene()) return;
    const node = event.target.closest('[data-node]');
    if (state.preview && node) { selectComponent(node.dataset.node, {additive: event.shiftKey}); toast('Apply or dismiss the preview before moving parts.'); return; }
    gesture = {identifier: node?.dataset.node, additive: event.shiftKey, startX: event.clientX, startY: event.clientY, camera: {...state.camera}, before: snapshotLayout(), moved: false};
    if (gesture.identifier) gesture.point = pointFor(gesture.identifier);
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!gesture) return;
    const deltaX = event.clientX - gesture.startX;
    const deltaY = event.clientY - gesture.startY;
    if (Math.abs(deltaX) + Math.abs(deltaY) > 4) gesture.moved = true;
    if (!gesture.moved) return;
    if (gesture.identifier) {
      setPoint(gesture.identifier, {x: gesture.point.x + deltaX / state.camera.scale, y: gesture.point.y + deltaY / state.camera.scale});
      renderDiagram();
    } else {
      state.camera.x = gesture.camera.x + deltaX;
      state.camera.y = gesture.camera.y + deltaY;
      updateCamera();
    }
  });
  const finish = (event) => {
    if (!gesture) return;
    if (gesture.identifier) {
      remember(gesture.before);
      selectComponent(gesture.identifier, {additive: gesture.additive});
      if (!gesture.moved && !gesture.additive && event.type === 'pointerup') openNodeMenu(gesture.identifier, event.clientX, event.clientY);
    }
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    gesture = null;
    canvas.classList.remove('dragging');
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', finish);
  canvas.addEventListener('dblclick', (event) => {
    const node = event.target.closest('[data-node]');
    if (!node) return;
    state.selected = node.dataset.node;
    state.isolated = !state.isolated;
    renderDiagram(); updateToolbar(); fitView();
  });
  canvas.addEventListener('keydown', (event) => {
    const node = event.target.closest('[data-node]');
    if (node && ['Enter', ' '].includes(event.key)) {
      event.preventDefault();
      const identifier = node.dataset.node;
      const bounds = node.getBoundingClientRect();
      selectComponent(identifier, {additive: event.shiftKey});
      if (!event.shiftKey) openNodeMenu(identifier, bounds.right, bounds.top, true);
    }
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    zoom(Math.exp(-event.deltaY * 0.0015), {x: event.clientX - bounds.left, y: event.clientY - bounds.top});
  }, {passive: false});
}

async function renderInspector() {
  const sequence = ++state.inspectorSequence;
  const part = component(state.selected);
  if (!part) return;
  const panel = select('#inspector-content');
  select('#inspector-title').textContent = part.name;
  select('#inspector-subtitle').textContent = part.category.toUpperCase() + ' / ' + part.id;
  for (const button of selectAll('[data-inspector-tab]')) {
    const active = button.dataset.inspectorTab === state.tab;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active);
  }
  panel.replaceChildren();
  if (state.tab === 'ai') { renderPartAI(panel, part); return; }
  if (state.tab === 'source') {
    if (!part.source) { panel.append(empty('Visual draft stage', 'This part has no implementation yet. Its connections and notes belong to your saved workspace.')); return; }
    panel.append(element('div', 'source-path', part.source), empty('Reading module…'));
    try {
      const source = await api(`/api/components/${encodeURIComponent(part.id)}/source`);
      if (sequence !== state.inspectorSequence) return;
      panel.replaceChildren(element('div', 'source-path', source.path));
      panel.append(action('Expand source ↗', () => showArtifact(source.path, source.content), 'source-link'));
      const listing = element('div', 'source-listing');
      const lines = source.content.split('\n');
      listing.append(element('pre', 'line-numbers', lines.map((line, index) => index + 1).join('\n')), codeBlock(source.content));
      panel.append(listing);
      if (source.truncated) panel.prepend(element('p', 'notice', 'Preview limited to the first 128 KiB.'));
    } catch (error) { if (sequence === state.inspectorSequence) panel.replaceChildren(empty('Source unavailable', error.message)); }
    return;
  }
  if (state.tab === 'data') {
    if (!state.run) { panel.append(empty('Select a saved run', 'Component data is read from local run snapshots.')); return; }
    const related = state.run.events.filter((event) => activeComponent(event) === part.id);
    panel.append(keyValue('Selected run', state.run.name), keyValue('Matching visible events', related.length));
    const event = currentEvent();
    if (event) {
      const detail = section('Selected recorded event');
      detail.append(badge(event.kind), codeBlock(event));
      panel.append(detail);
    }
    if (part.category === 'storage') {
      const records = section('Saved records');
      records.append(keyValue('Visible knowledge nodes', state.run.nodes.length), keyValue('Artifact files', state.run.artifacts.length));
      panel.append(records);
    }
    if (!related.length) panel.append(empty('No mapped events in this snapshot', 'The architectural component exists independently of the selected run.'));
    for (const item of related.slice(-12).reverse()) {
      const row = action('', () => setEvent(state.run.events.indexOf(item)), 'event-data-button');
      row.append(element('strong', '', human(item.kind)), element('span', '', timestamp(item.at, true)));
      panel.append(row);
    }
    return;
  }
  const intro = section('Component');
  intro.append(element('p', 'component-description', part.description));
  if (part.source) intro.append(action(part.source.split('/').pop() + ' ↗', () => { state.tab = 'source'; renderInspector(); }, 'source-link'));
  else intro.append(badge('visual draft'));
  panel.append(intro);
  for (const [label, values] of [['Inputs', part.inputs], ['Outputs', part.outputs]]) {
    const portSection = section(label);
    for (const value of values) {
      const port = element('div', 'port-row');
      port.append(element('span', 'port-dot'), element('span', '', value));
      portSection.append(port);
    }
    panel.append(portSection);
  }
  const connected = section('Connected parts');
  for (const edge of scene().edges.filter((edge) => edge.source === part.id || edge.target === part.id)) {
    const other = edge.source === part.id ? edge.target : edge.source;
    const link = action('', () => selectComponent(other), 'connection-link');
    link.append(element('span', '', `${edge.source === part.id ? '↗' : '↙'} ${component(other).name}`), element('small', '', edge.label));
    connected.append(link);
  }
  panel.append(connected);
  const invariants = section('Design constraints');
  for (const item of part.invariants) invariants.append(element('p', 'invariant', item));
  panel.append(invariants);
  if (part.id.startsWith('research-')) {
    const defaults = section('Current source defaults');
    for (const [role, model] of Object.entries(state.overview.defaults.research || {})) defaults.append(keyValue(role, model));
    panel.append(defaults);
  }
  const notes = section('Workbench notes');
  const input = element('textarea', 'component-notes');
  input.placeholder = 'Design notes for this component…';
  input.maxLength = 4000;
  input.rows = 3;
  input.value = effectiveWorkspace().notes[part.id] || '';
  input.disabled = Boolean(state.preview);
  input.setAttribute('aria-label', `Notes for ${part.name}`);
  input.addEventListener('input', () => { state.notes[part.id] = input.value; persist(); });
  notes.append(input, element('small', 'muted', 'Saved in this browser and exported with your workspace.'));
  panel.append(notes);
}

function metric(label, value, detail = '') {
  const card = element('div', 'metric-card');
  card.append(element('span', 'metric-label', label), element('strong', 'metric-value', value));
  if (detail) card.append(element('small', 'metric-detail', detail));
  return card;
}

function table(headers, rows) {
  const wrap = element('div', 'table-wrap');
  const grid = element('table');
  const head = element('thead');
  const heading = element('tr');
  for (const title of headers) heading.append(element('th', '', title));
  head.append(heading);
  const body = element('tbody');
  for (const row of rows) {
    const line = element('tr');
    for (const value of row) {
      const cell = element('td');
      if (value instanceof Node) cell.append(value); else cell.textContent = value ?? '—';
      line.append(cell);
    }
    body.append(line);
  }
  grid.append(head, body);
  wrap.append(grid);
  return wrap;
}

function renderDashboard() {
  const content = select('#dashboard-content');
  content.replaceChildren();
  if (!state.overview) return;
  const heading = element('div', 'dashboard-heading');
  const title = element('div');
  title.append(element('div', 'eyebrow', 'OBSERVATORY / LOCAL SNAPSHOTS'), element('h1', '', 'Inside the harness'), element('p', 'muted', 'Inspect what ran, what moved, and what the saved evidence actually records.'));
  heading.append(title, badge('read only'));
  content.append(heading);
  const counts = element('div', 'stat-grid');
  counts.append(metric('Saved runs', number(state.overview.counts.runs), `${state.overview.counts.mission_runs} missions · ${state.overview.counts.research_runs} research`), metric('Components', number(state.overview.counts.components), 'Two independent architectures'), metric('Recorded tokens', compact(state.overview.runs.reduce((total, run) => total + run.tokens, 0)), 'Known usage across visible runs'), metric('Completed runs', number(state.overview.runs.filter((run) => /completed|accepted|passed/.test(run.status)).length), 'Recorded status, not a quality score'));
  content.append(counts);
  const runsSection = section('Saved run inventory');
  const controls = element('div', 'inventory-controls');
  const search = element('input', 'run-search');
  search.placeholder = 'Filter runs by name, model, or status';
  search.setAttribute('aria-label', 'Filter saved runs');
  const filter = element('select');
  filter.setAttribute('aria-label', 'Filter run type');
  for (const [value, label] of [['all', 'All run types'], ['mission', 'Missions'], ['research', 'Research']]) { const option = element('option', '', label); option.value = value; filter.append(option); }
  controls.append(search, filter);
  const inventory = element('div', 'run-inventory');
  function drawInventory() {
    const query = search.value.toLowerCase();
    const runs = state.overview.runs.filter((run) => (filter.value === 'all' || run.kind === filter.value) && `${run.name} ${run.model} ${run.status}`.toLowerCase().includes(query));
    const rows = runs.map((run) => {
      const button = action(run.name, () => loadRun(run.id), 'run-name' + (run.id === state.runId ? ' selected' : ''));
      button.title = run.relative_path;
      return [button, badge(run.status), human(run.kind), number(run.workers), compact(run.tokens), timestamp(run.updated_at)];
    });
    inventory.replaceChildren(rows.length ? table(['Run', 'Status', 'Workflow', 'Model nodes', 'Recorded tokens', 'Updated'], rows) : empty('No matching runs'));
  }
  search.addEventListener('input', drawInventory);
  filter.addEventListener('change', drawInventory);
  drawInventory();
  runsSection.append(controls, inventory);
  content.append(runsSection);
  if (!state.run) { content.append(empty('Select a run to inspect its records')); return; }
  const run = state.run;
  const detail = element('div', 'dashboard-run-detail');
  detail.id = 'selected-run-detail';
  const runHeading = element('div', 'run-detail-heading');
  runHeading.append(element('h2', '', run.name), badge(run.status), action('Inspect configuration', () => showArtifact('Saved run configuration', JSON.stringify(run.config, null, 2))));
  detail.append(runHeading, element('p', 'source-path', run.relative_path));
  if (run.warnings.length) detail.append(element('p', 'notice', run.warnings.join(' · ')));
  if (run.metrics.usage?.usage_complete === false) detail.append(element('p', 'notice', 'Usage is incomplete. Token figures show only recorded totals; unresolved requests may have additional usage.'));
  const stats = element('div', 'stat-grid');
  stats.append(metric(run.kind === 'research' ? 'Model roles' : 'Workers', number(run.workers.length), run.kind === 'research' ? `${run.config.workers || 'Up to 3'} parallel workers + head / QC` : run.provider), metric('Provider requests', number(run.requests), 'Recorded attempts'), metric('Recorded tokens', compact(run.tokens), short(run.model, 42)), metric('Peer deliveries', number(run.deliveries.length), run.kind === 'research' ? 'Research shares artifacts directly' : `${number(run.metrics.deliveries)} saved delivery rows`));
  detail.append(stats);
  if (run.phases.length) {
    const phases = section('Research phase history');
    const strip = element('div', 'phase-strip');
    for (const phase of run.phases) {
      const chip = element('div', 'phase-chip');
      chip.append(element('small', '', 'ROUND ' + phase.round), element('strong', '', human(phase.name)), badge(phase.status), element('small', 'muted', short(phase.model, 40)));
      strip.append(chip);
    }
    phases.append(strip);
    detail.append(phases);
  }
  const workers = section(run.kind === 'research' ? 'Model role states' : 'Worker states');
  workers.append(table(['Worker', 'Recorded state', 'Model'], run.workers.map((worker) => [worker.id, badge(worker.state), worker.model])));
  detail.append(workers);
  if (run.deliveries.length) {
    const receipts = section('Peer delivery ledger');
    const counts = run.deliveries.reduce((totals, row) => { totals[row.state] = (totals[row.state] || 0) + 1; return totals; }, {});
    const strip = element('div', 'receipt-counts');
    for (const [status, count] of Object.entries(counts)) strip.append(badge(`${count} ${status}`));
    receipts.append(strip, element('p', 'muted', 'Accepted steering, inbox delivery, acknowledgement, and incorporation are distinct records.'));
    receipts.append(table(['From', 'To', 'State', 'Event'], run.deliveries.map((row) => [row.sender, row.recipient, badge(row.state), short(row.event_id || row.id, 16)])));
    detail.append(receipts);
  }
  const comparison = section('Compare recorded runs');
  const comparisonSelect = element('select');
  comparisonSelect.setAttribute('aria-label', 'Compare selected run with');
  const placeholder = element('option', '', 'Choose another saved run…');
  placeholder.value = '';
  comparisonSelect.append(placeholder);
  for (const other of state.overview.runs.filter((item) => item.id !== run.id)) { const option = element('option', '', other.name); option.value = other.id; comparisonSelect.append(option); }
  const comparisonBody = element('div', 'comparison-body');
  function compare() {
    state.comparisonId = comparisonSelect.value;
    const other = state.overview.runs.find((item) => item.id === state.comparisonId);
    if (!other) { comparisonBody.replaceChildren(); return; }
    const first = state.overview.runs.find((item) => item.id === run.id);
    comparisonBody.replaceChildren(table(['Recorded metric', run.name, other.name], [['Workflow', first.kind, other.kind], ['Status', first.status, other.status], ['Workers', number(first.workers), number(other.workers)], ['Provider requests', number(first.requests), number(other.requests)], ['Known tokens', number(first.tokens), number(other.tokens)], ['Knowledge records / requests', number(first.events), number(other.events)], ['Deliveries', number(first.deliveries), number(other.deliveries)]]), element('p', 'muted', 'Run configurations and evidence can differ. These counts are not a controlled quality benchmark.'));
  }
  comparisonSelect.value = state.comparisonId || '';
  comparisonSelect.addEventListener('change', compare);
  comparison.append(comparisonSelect, comparisonBody);
  compare();
  detail.append(comparison);
  const artifacts = section(`Artifacts / ${run.artifacts.length}`);
  const artifactSearch = element('input');
  artifactSearch.placeholder = 'Filter artifact files';
  artifactSearch.setAttribute('aria-label', 'Filter artifacts');
  const artifactList = element('div', 'artifact-list');
  function drawArtifacts() {
    const query = artifactSearch.value.toLowerCase();
    artifactList.replaceChildren();
    for (const artifact of run.artifacts.filter((item) => item.name.toLowerCase().includes(query))) {
      const button = action('', () => openArtifact(artifact), 'artifact-button');
      button.append(element('span', '', artifact.name), element('small', 'muted', `${compact(artifact.size)} B · ${artifact.kind}`));
      artifactList.append(button);
    }
    if (!artifactList.childElementCount) artifactList.append(empty('No matching artifacts'));
  }
  artifactSearch.addEventListener('input', drawArtifacts);
  artifacts.append(artifactSearch, artifactList);
  drawArtifacts();
  detail.append(artifacts);
  content.append(detail);
}

function poincareDistance(left, right) {
  const leftNorm = left[0] ** 2 + left[1] ** 2;
  const rightNorm = right[0] ** 2 + right[1] ** 2;
  if (leftNorm >= 1 || rightNorm >= 1) return null;
  const squared = (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2;
  return Math.acosh(Math.max(1, 1 + 2 * squared / ((1 - leftNorm) * (1 - rightNorm))));
}

function renderGeometry() {
  const content = select('#geometry-content');
  content.replaceChildren();
  const heading = element('div', 'dashboard-heading');
  const title = element('div');
  title.append(element('div', 'eyebrow', 'SPATIAL INSPECTOR / CURVATURE −1'), element('h1', '', 'Knowledge geometry'), element('p', 'muted', 'Saved hierarchy coordinates in the Poincaré disk. Select a point to inspect its neighborhood.'));
  heading.append(title, badge('saved coordinates'));
  content.append(heading);
  if (!state.run || state.run.kind === 'research') {
    content.append(empty(state.run ? 'This research run uses artifacts, not geometric routing.' : 'Select a mission run with saved coordinates.', 'The mission architecture stores hierarchy-derived points. The research loop is a separate workflow.'));
    const missions = state.overview?.runs.filter((run) => run.kind === 'mission') || [];
    if (missions.length) content.append(action('Inspect a saved mission', () => loadRun((missions.find((run) => run.name === 'example-run') || missions[0]).id)));
    return;
  }
  const points = state.run.nodes.filter((node) => Array.isArray(node.position) && node.position.length === 2 && node.position.every(Number.isFinite) && Math.hypot(...node.position) < 1);
  if (!points.length) { content.append(empty('No valid coordinates in this snapshot')); return; }
  const selected = points.find((node) => node.id === state.geometrySelection) || points[0];
  state.geometrySelection = selected.id;
  const grid = element('div', 'geometry-grid');
  const plot = element('div', 'geometry-plot');
  const disk = svgElement('svg', {viewBox: '0 0 620 620', role: 'img', 'aria-label': `${points.length} saved knowledge points in a Poincaré disk`});
  disk.append(svgElement('circle', {cx: 310, cy: 310, r: 268, class: 'disk-boundary'}));
  for (const radius of [0.25, 0.5, 0.75]) disk.append(svgElement('circle', {cx: 310, cy: 310, r: 268 * radius, class: 'disk-ring'}));
  disk.append(svgElement('line', {x1: 42, y1: 310, x2: 578, y2: 310, class: 'disk-axis'}), svgElement('line', {x1: 310, y1: 42, x2: 310, y2: 578, class: 'disk-axis'}));
  for (const label of [{x: 588, y: 315, text: '+1'}, {x: 18, y: 315, text: '−1'}, {x: 304, y: 28, text: '+1'}, {x: 304, y: 606, text: '−1'}, {x: 320, y: 329, text: '0'}]) disk.append(svgElement('text', {x: label.x, y: label.y, class: 'disk-label'}, label.text));
  const authors = [...new Set(points.map((node) => node.author))];
  const palette = ['#59d9ce', '#edbc72', '#a699ed', '#87aafa', '#e38b99', '#8ba0b8'];
  for (const node of points) {
    const marker = svgElement('g', {class: 'geometry-point' + (node.id === selected.id ? ' selected' : ''), role: 'button', tabindex: '0', 'aria-label': `${node.author}: ${short(node.claim, 100)}`});
    const positionX = 310 + node.position[0] * 268;
    const positionY = 310 - node.position[1] * 268;
    const color = palette[authors.indexOf(node.author) % palette.length];
    marker.append(svgElement('title', {}, `${node.author} · ${node.scope_path.join(' / ')}\n${node.claim}`));
    marker.append(svgElement('circle', {cx: positionX, cy: positionY, r: node.id === selected.id ? 13 : 9, fill: color, 'fill-opacity': 0.12, stroke: node.id === selected.id ? color : 'none'}));
    marker.append(svgElement('circle', {cx: positionX, cy: positionY, r: 4.5, fill: color}));
    marker.addEventListener('click', () => { state.geometrySelection = node.id; renderGeometry(); });
    marker.addEventListener('keydown', (event) => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); state.geometrySelection = node.id; renderGeometry(); } });
    disk.append(marker);
  }
  plot.append(disk);
  const legend = element('div', 'geometry-legend');
  authors.forEach((author, index) => { const item = element('span', 'legend-item'); const dot = element('span', 'component-dot'); dot.style.background = palette[index % palette.length]; item.append(dot, document.createTextNode(author)); legend.append(item); });
  plot.append(legend, element('p', 'muted', `${points.length} visible coordinates · overlapping points can share the same hierarchy path`));
  const inspector = element('div', 'geometry-inspector');
  const details = section('Selected knowledge record');
  details.append(badge(selected.type), element('p', 'geometry-claim', selected.claim), keyValue('Author', selected.author), keyValue('Position', selected.position.map((value) => value.toFixed(5)).join(', ')), element('p', 'source-path', selected.scope_path.join(' / ')), action('Inspect record', () => showArtifact('Saved knowledge record', JSON.stringify(selected, null, 2))));
  inspector.append(details);
  const nearest = points.filter((node) => node.id !== selected.id).map((node) => ({...node, distance: poincareDistance(selected.position, node.position)})).sort((left, right) => left.distance - right.distance).slice(0, 8);
  const neighborhood = section('Nearest saved points');
  neighborhood.append(element('p', 'muted', 'Exact local reference distances, computed from saved coordinates.'));
  for (const node of nearest) {
    const button = action('', () => { state.geometrySelection = node.id; renderGeometry(); }, 'neighbor-row');
    const copy = element('span');
    copy.append(element('strong', '', node.author), element('small', '', short(node.claim, 56)));
    button.append(copy, element('span', 'distance-value', node.distance.toFixed(4)));
    neighborhood.append(button);
  }
  inspector.append(neighborhood);
  grid.append(plot, inspector);
  content.append(grid, element('p', 'notice', 'Coordinates describe known hierarchy structure. Proximity is not a truth score or a learned semantic similarity. This view does not query or change the HyperspaceDB service.'));
}

function showArtifact(name, content) {
  state.artifactSequence += 1;
  select('#artifact-title').textContent = name;
  select('#artifact-content').textContent = content;
  const dialog = select('#artifact-dialog');
  if (!dialog.open) dialog.showModal();
}

async function openArtifact(artifact) {
  showArtifact(artifact.name, 'Reading saved artifact…');
  const sequence = state.artifactSequence;
  const runId = state.runId;
  try {
    const record = await api(`/api/runs/${runId}/artifacts/${artifact.id}`);
    if (sequence !== state.artifactSequence || runId !== state.runId || !select('#artifact-dialog').open) return;
    showArtifact(record.name + (record.truncated ? ' · preview truncated' : ''), record.content);
  } catch (error) {
    if (sequence === state.artifactSequence && runId === state.runId && select('#artifact-dialog').open) select('#artifact-content').textContent = error.message;
  }
}

function renderRunSelect() {
  const dropdown = select('#run-select');
  dropdown.replaceChildren();
  if (!state.overview.runs.length) { const option = element('option', '', 'No saved runs found'); option.value = ''; dropdown.append(option); dropdown.disabled = true; return; }
  dropdown.disabled = false;
  for (const kind of ['mission', 'research']) {
    const group = element('optgroup');
    group.label = kind === 'mission' ? 'Mission snapshots' : 'Research snapshots';
    for (const run of state.overview.runs.filter((item) => item.kind === kind)) {
      const option = element('option', '', `${run.name} · ${human(run.status)}`);
      option.value = run.id;
      group.append(option);
    }
    if (group.childElementCount) dropdown.append(group);
  }
  dropdown.value = state.runId || '';
}

async function loadRun(identifier, {quiet = false, preserveScene = false} = {}) {
  if (!identifier) return;
  const sequence = ++state.runSequence;
  state.pendingRunId = identifier;
  stopPlayback();
  if (!quiet) select('#run-status').textContent = 'Reading snapshot…';
  try {
    const run = await api(`/api/runs/${encodeURIComponent(identifier)}`);
    if (sequence !== state.runSequence) return;
    const sameRun = state.runId === identifier;
    const priorEvent = currentEvent()?.id;
    state.run = run;
    state.runId = identifier;
    state.geometrySelection = sameRun ? state.geometrySelection : null;
    state.eventIndex = quiet && priorEvent ? Math.max(0, run.events.findIndex((event) => event.id === priorEvent)) : run.events.length - 1;
    if (!quiet && !preserveScene && run.kind !== state.sceneId) switchScene(run.kind);
    renderRunSelect();
    renderTimeline();
    renderDiagram();
    if (state.tab === 'data') renderInspector();
    if (state.view === 'dashboard') renderDashboard();
    if (state.view === 'geometry') renderGeometry();
    select('#connection-status').textContent = 'Local reader connected';
  } catch (error) {
    if (sequence === state.runSequence) {
      showError(error);
      select('#run-status').textContent = 'Snapshot unavailable';
      renderRunSelect();
    }
  } finally { if (sequence === state.runSequence) state.pendingRunId = null; }
}

function renderTimeline() {
  const run = state.run;
  const list = select('#event-list');
  list.replaceChildren();
  const events = run?.events || [];
  select('#run-status').replaceChildren(badge(run?.status || 'No run selected'));
  select('#timeline-slider').max = Math.max(0, events.length - 1);
  select('#timeline-slider').value = Math.max(0, state.eventIndex);
  select('#timeline-slider').disabled = !events.length;
  select('#play-button').disabled = !events.length;
  select('#timeline-position').textContent = events.length ? `${state.eventIndex + 1} / ${events.length}` : '0 / 0';
  select('#run-metrics').replaceChildren();
  if (run) select('#run-metrics').append(element('span', '', `${run.workers.length} ${run.kind === 'research' ? 'model roles' : 'workers'}`), element('span', '', `${compact(run.tokens)} recorded tokens`), element('span', '', `${run.deliveries.length} visible deliveries`), element('span', '', `Snapshot ${timestamp(run.updated_at)}`));
  if (!events.length) { list.append(empty('No recorded events', 'Select another run, or refresh after the harness saves new data.')); return; }
  for (const [index, event] of events.entries()) {
    const row = action('', () => setEvent(index), 'event-row' + (index === state.eventIndex ? ' active' : ''));
    row.dataset.eventIndex = index;
    row.setAttribute('aria-label', `Event ${index + 1}: ${human(event.kind)}`);
    row.append(element('span', 'event-sequence', String(index + 1).padStart(3, '0')), element('time', 'event-time', timestamp(event.at, true)), element('span', 'event-kind', human(event.kind)), element('span', 'event-agent', event.agent || 'coordinator'), element('span', 'event-summary', short(event.summary, 120)));
    list.append(row);
  }
  scrollToEvent();
}

function scrollToEvent() {
  const list = select('#event-list');
  const active = list.querySelector('.active');
  if (active) list.scrollTop = active.offsetTop - list.offsetTop - list.clientHeight / 2 + active.offsetHeight / 2;
}

function setEvent(index) {
  if (!state.run?.events.length) return;
  state.eventIndex = Math.min(state.run.events.length - 1, Math.max(0, Number(index)));
  select('#timeline-slider').value = state.eventIndex;
  select('#timeline-position').textContent = `${state.eventIndex + 1} / ${state.run.events.length}`;
  for (const row of selectAll('[data-event-index]')) row.classList.toggle('active', Number(row.dataset.eventIndex) === state.eventIndex);
  renderDiagram();
  if (state.tab === 'data') renderInspector();
  scrollToEvent();
}

function stopPlayback() {
  if (state.playing) clearInterval(state.playing);
  state.playing = null;
  select('#play-button').textContent = '▶';
  select('#play-button').setAttribute('aria-label', 'Play recorded events');
}

function togglePlayback() {
  if (state.playing) { stopPlayback(); return; }
  if (!state.run?.events.length) return;
  if (state.eventIndex >= state.run.events.length - 1) setEvent(0);
  select('#play-button').textContent = 'Ⅱ';
  select('#play-button').setAttribute('aria-label', 'Pause recorded events');
  state.playing = setInterval(() => {
    if (state.eventIndex >= state.run.events.length - 1) { stopPlayback(); return; }
    setEvent(state.eventIndex + 1);
  }, 650);
}

async function refresh(initial = false) {
  if (state.refreshing) return;
  const runSequence = state.runSequence;
  state.refreshing = true;
  select('#refresh-button').disabled = true;
  try {
    const overview = await api('/api/overview');
    state.overview = overview;
    try { validateWorkspace(workspace()); }
    catch {
      state.drafts = {};
      for (const [sceneId, positions] of Object.entries(state.positions)) {
        const known = new Set(baseScene(sceneId)?.nodes.map(node => node.id) || []);
        for (const identifier of Object.keys(positions)) if (!known.has(identifier)) delete positions[identifier];
      }
      const known = new Set(catalog().components.map(part => part.id));
      for (const identifier of Object.keys(state.notes)) if (!known.has(identifier)) delete state.notes[identifier];
      for (const identifier of Object.keys(state.aiParts)) if (!known.has(identifier)) delete state.aiParts[identifier];
      state.selected = state.sceneId === 'research' ? 'research-plan' : 'coordinator';
      state.selection = new Set([state.selected]);
      toast('An incompatible saved draft was cleared; source-backed layout and notes were kept.');
    }
    reconcileSelection();
    select('#error-banner').hidden = true;
    select('#last-updated').textContent = 'Read ' + timestamp(overview.generated_at, true);
    select('#connection-status').textContent = 'Local reader connected';
    if (overview.diagnostics.length) { select('#error-banner').hidden = false; select('#error-banner').textContent = overview.diagnostics.map((item) => typeof item === 'string' ? item : item.detail || item.title).join(' · '); }
    const selected = overview.runs.find((run) => run.id === state.runId);
    const fallback = overview.runs.find((run) => run.name === 'example-run') || overview.runs[0];
    renderExplorer(); renderDiagram(); renderInspector(); renderRunSelect();
    if (runSequence === state.runSequence && !state.pendingRunId) {
      if (selected || fallback) await loadRun((selected || fallback).id, {quiet: !initial, preserveScene: true});
      else { state.run = null; state.runId = null; renderTimeline(); }
    }
    if (state.view === 'dashboard') renderDashboard();
    if (initial) requestAnimationFrame(fitView);
    if (!initial) toast('Saved run data refreshed.');
  } catch (error) { showError(error); }
  finally { state.refreshing = false; select('#refresh-button').disabled = false; }
}

function validateDrafts(drafts) {
  if (!drafts || typeof drafts !== 'object' || Array.isArray(drafts)) throw new Error('Invalid workflow drafts.');
  const draftIds = new Set();
  for (const [sceneId, draft] of Object.entries(drafts)) {
    if (!['mission', 'research'].includes(sceneId) || !validFields(draft, ['nodes', 'edges', 'parameters', 'hidden']) || !Array.isArray(draft.nodes) || !Array.isArray(draft.edges) || draft.nodes.length > 24 || draft.edges.length > 80) throw new Error('Invalid workflow draft.');
    const base = baseScene(sceneId);
    const identifiers = new Set(base?.nodes.map((node) => node.id) || []);
    if (Object.hasOwn(draft, 'hidden') && (!Array.isArray(draft.hidden) || draft.hidden.length > 24 || new Set(draft.hidden).size !== draft.hidden.length || draft.hidden.some(identifier => typeof identifier !== 'string' || (base && !identifiers.has(identifier))))) throw new Error('Invalid hidden draft components.');
    for (const node of draft.nodes) {
      if (!validFields(node, ['id', 'name', 'description', 'category', 'x', 'y']) || !/^draft-[a-z0-9][a-z0-9-]{0,53}$/.test(node.id) || draftIds.has(node.id) || typeof node.name !== 'string' || !node.name.trim() || node.name.length > 80 || typeof node.description !== 'string' || node.description.length > 1600 || !Object.hasOwn(COLORS, node.category) || !Number.isFinite(node.x) || !Number.isFinite(node.y) || Math.abs(node.x) > 20000 || Math.abs(node.y) > 20000) throw new Error('Invalid draft component.');
      draftIds.add(node.id);
      identifiers.add(node.id);
    }
    const edges = new Set();
    for (const edge of draft.edges) {
      if (!validFields(edge, ['source', 'target', 'label', 'kind']) || !/^[a-z][a-z0-9-]{0,60}$/.test(edge.source) || !/^[a-z][a-z0-9-]{0,60}$/.test(edge.target) || edge.source === edge.target || !['control', 'data', 'feedback'].includes(edge.kind) || typeof edge.label !== 'string' || edge.label.length > 160 || (base && (!identifiers.has(edge.source) || !identifiers.has(edge.target)))) throw new Error('Invalid draft connection.');
      const key = edge.source + ':' + edge.target;
      if (edges.has(key)) throw new Error('Duplicate draft connection.');
      edges.add(key);
    }
    if (Object.hasOwn(draft, 'parameters') && !validFields(draft.parameters, [])) throw new Error('Draft execution settings are not supported.');
  }
}

function dismissPreview(redraw = true) {
  if (!state.preview) return;
  state.preview = null;
  if (select('#draft-preview-bar')) select('#draft-preview-bar').hidden = true;
  reconcileSelection();
  if (redraw) { renderExplorer(); renderDiagram(); renderInspector(); updateToolbar(); }
}

function previewProposal(proposal, sceneId) {
  if (!baseScene(sceneId)) throw new Error('This proposal refers to an unknown architecture.');
  if (!proposal || typeof proposal.title !== 'string' || !Array.isArray(proposal.operations) || !proposal.operations.length || proposal.operations.length > 12) throw new Error('This response has no valid workflow changes to preview.');
  dismissPreview(false);
  if (state.sceneId !== sceneId) switchScene(sceneId);
  const candidate = JSON.parse(JSON.stringify(workspace()));
  candidate.drafts[sceneId] ||= {nodes: [], edges: structuredClone(baseScene(sceneId).edges), parameters: {}};
  candidate.positions[sceneId] ||= {};
  const draft = candidate.drafts[sceneId];
  const identifiers = new Set([...baseScene(sceneId).nodes, ...draft.nodes].map((node) => node.id));
  function requirePart(identifier) {
    if (typeof identifier !== 'string' || !identifiers.has(identifier)) throw new Error('A proposed change refers to an unavailable component.');
  }
  function requirePoint(operation) {
    if (!Number.isFinite(operation.x) || !Number.isFinite(operation.y) || operation.x < 0 || operation.x > 2400 || operation.y < 0 || operation.y > 1600) throw new Error('A proposed position is outside the design canvas.');
  }
  for (const operation of proposal.operations) {
    if (!operation || typeof operation !== 'object') throw new Error('Invalid proposed change.');
    if (operation.type === 'add_component') {
      if (!/^draft-[a-z0-9][a-z0-9-]{0,53}$/.test(operation.component_id) || identifiers.has(operation.component_id)) throw new Error('Draft stage IDs must be new and unique.');
      requirePoint(operation);
      draft.nodes.push({id: operation.component_id, name: operation.name, description: operation.description, category: operation.category, x: operation.x, y: operation.y});
      identifiers.add(operation.component_id);
    } else if (operation.type === 'move_component') {
      requirePart(operation.component_id);
      requirePoint(operation);
      candidate.positions[sceneId][operation.component_id] = {x: operation.x, y: operation.y};
    } else if (operation.type === 'set_note') {
      requirePart(operation.component_id);
      if (typeof operation.text !== 'string' || operation.text.length > 4000) throw new Error('The proposed note is too long.');
      candidate.notes[operation.component_id] = operation.text;
    } else if (operation.type === 'connect') {
      requirePart(operation.source); requirePart(operation.target);
      draft.edges = draft.edges.filter((edge) => edge.source !== operation.source || edge.target !== operation.target);
      draft.edges.push({source: operation.source, target: operation.target, label: operation.label, kind: operation.kind});
    } else if (operation.type === 'disconnect') {
      requirePart(operation.source); requirePart(operation.target);
      if (!draft.edges.some((edge) => edge.source === operation.source && edge.target === operation.target)) throw new Error('A connection to remove no longer exists. Ask for an updated proposal.');
      draft.edges = draft.edges.filter((edge) => edge.source !== operation.source || edge.target !== operation.target);
    } else if (operation.type === 'remove_component') {
      requirePart(operation.component_id);
      if (!operation.component_id.startsWith('draft-')) throw new Error('Existing harness components cannot be removed. Only draft stages can be deleted.');
      draft.nodes = draft.nodes.filter((node) => node.id !== operation.component_id);
      draft.edges = draft.edges.filter((edge) => edge.source !== operation.component_id && edge.target !== operation.component_id);
      delete candidate.positions[sceneId][operation.component_id];
      delete candidate.notes[operation.component_id];
      if (candidate.ai_parts) delete candidate.ai_parts[operation.component_id];
      identifiers.delete(operation.component_id);
    } else throw new Error('This type of workflow change is not supported.');
  }
  validateWorkspace(candidate);
  showDraftPreview(candidate, sceneId, proposal.title, proposal.operations);
}

function showDraftPreview(candidate, sceneId, title, operations) {
  state.preview = {candidate, sceneId, title, operations, baseline: snapshotLayout()};
  state.isolated = false;
  switchView('assembly');
  select('#draft-preview-bar').hidden = false;
  select('#draft-preview-title').textContent = title;
  select('#draft-preview-summary').textContent = `${operations.length} proposed changes · visual draft only`;
  reconcileSelection();
  renderExplorer(); renderDiagram(); renderInspector(); updateToolbar(); requestAnimationFrame(fitView);
}

function applyPreview() {
  const preview = state.preview;
  if (!preview) return;
  if (preview.baseline !== snapshotLayout()) { dismissPreview(); toast('Your workspace changed. Preview the proposal again before applying.'); return; }
  const before = snapshotLayout();
  state.positions = preview.candidate.positions;
  state.notes = preview.candidate.notes;
  state.drafts = preview.candidate.drafts;
  state.aiParts = preview.candidate.ai_parts || {};
  dismissPreview(false);
  remember(before);
  renderExplorer(); renderDiagram(); renderInspector(); updateToolbar();
  window.dispatchEvent(new CustomEvent('workbench:draft-applied', {detail: {title: preview.title}}));
  toast('Visual draft applied. Undo restores the previous workflow.');
}

function assistantContext(selectedText = '') {
  const draft = state.drafts[state.sceneId];
  const graph = {nodes: [...(baseScene()?.nodes || []).filter(node => !draft?.hidden?.includes(node.id)), ...(draft?.nodes || [])]};
  const partIds = [...state.selection].filter((identifier) => graph?.nodes.some((node) => node.id === identifier)).slice(0, 8);
  return {
    selection: {scene: state.sceneId, component_ids: partIds, run_id: state.run?.kind === state.sceneId ? state.runId : null, event_id: state.run?.kind === state.sceneId ? currentEvent()?.id || null : null, selected_text: String(selectedText || '').slice(0, 4000)},
    workspace: JSON.parse(JSON.stringify(workspace())),
    display: {scene: baseScene()?.name || state.sceneId, parts: partIds.map((identifier) => ({id: identifier, name: component(identifier)?.name || identifier})), run: state.run?.kind === state.sceneId ? state.run.name : null, event: state.run?.kind === state.sceneId ? currentEvent()?.kind : null},
  };
}

function askSelection(selectedText = '', prompt = '') {
  window.dispatchEvent(new CustomEvent('workbench:ask', {detail: {context: assistantContext(selectedText), prompt}}));
}

function renderPartAI(panel, part) {
  const profile = effectiveWorkspace().ai_parts?.[part.id] || {enabled: true, primary_model: '', fallback_model: '', max_steps: 10, temperature: 0.2};
  const header = section('AI orchestration');
  header.append(element('span', 'draft-badge', 'VISUAL CONFIGURATION'));
  header.append(element('p', 'ai-part-boundary', 'Design this part’s AI behavior. These values belong to the visual draft; they do not change the assistant connection or run the harness.'));
  const form = element('form', 'ai-part-form');
  const enabledLabel = element('label', 'ai-part-toggle');
  const enabled = element('input');
  enabled.type = 'checkbox';
  enabled.id = 'part-ai-enabled';
  enabled.checked = profile.enabled;
  enabledLabel.append(element('span', '', 'Enabled in draft'), enabled);
  form.append(enabledLabel);
  const controls = {enabled};
  for (const field of [
    {key: 'primary_model', label: 'Primary model', type: 'text', placeholder: 'Use workspace AI'},
    {key: 'fallback_model', label: 'Fallback model', type: 'text', placeholder: 'None planned'},
    {key: 'max_steps', label: 'Max steps', type: 'number', min: 1, max: 100, step: 1},
    {key: 'temperature', label: 'Temperature', type: 'number', min: 0, max: 2, step: 0.1},
  ]) {
    const label = element('label', 'ai-part-field', field.label);
    const input = element('input');
    input.id = 'part-ai-' + field.key.replaceAll('_', '-');
    input.type = field.type;
    input.value = profile[field.key];
    if (field.type === 'text') { input.maxLength = 200; input.placeholder = field.placeholder; }
    else { input.min = field.min; input.max = field.max; input.step = field.step; input.required = true; }
    label.htmlFor = input.id;
    form.append(label, input);
    controls[field.key] = input;
  }
  const submit = element('button', 'primary-button', 'Preview part settings');
  submit.id = 'part-ai-preview';
  submit.type = 'submit';
  submit.disabled = Boolean(state.preview);
  form.append(submit);
  for (const input of Object.values(controls)) input.disabled = Boolean(state.preview);
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (state.preview) return;
    const candidate = structuredClone(workspace());
    candidate.ai_parts ||= {};
    candidate.ai_parts[part.id] = {enabled: enabled.checked, primary_model: controls.primary_model.value.trim(), fallback_model: controls.fallback_model.value.trim(), max_steps: Number(controls.max_steps.value), temperature: Number(controls.temperature.value)};
    candidate.drafts[state.sceneId] ||= {nodes: [], edges: structuredClone(baseScene().edges), parameters: {}};
    try {
      validateWorkspace(candidate);
      closeNodeMenu();
      showDraftPreview(candidate, state.sceneId, `AI design settings · ${part.name}`, [{type: 'configure_ai', component_id: part.id}]);
    } catch (error) { toast(error.message); }
  });
  header.append(form, element('p', 'ai-part-boundary', 'Fallback and step limits are design notes, not active routing. Apply the preview to save; Undo restores the previous profile.'));
  header.append(action('Open AI provider settings ↗', () => switchView('settings'), 'source-link'));
  panel.append(header);
  const actions = section('Actions for this part');
  actions.append(action('Ask about this part', () => askSelection(), 'ask-component-button'));
  actions.append(action('Explain this part', () => askSelection('', 'Explain this part’s responsibilities and connections using the attached context.'), 'source-link'));
  panel.append(actions);
}

function openNodeAIMenu() {
  const part = component(menuComponent);
  if (!part) return;
  const submenu = select('#node-ai-menu');
  submenu.replaceChildren();
  const heading = element('div', 'node-menu-heading');
  heading.append(element('h2', '', part.name + ' · AI'));
  const close = action('×', () => closeNodeAIMenu(true), 'icon-button');
  close.setAttribute('aria-label', 'Close AI actions');
  heading.append(close);
  submenu.append(heading);
  const actions = element('div', 'node-menu-actions');
  const prompts = [
    ['ask', 'Ask', '↗', ''],
    ['inspect', 'Inspect', '⌕', null],
    ['refactor', 'Refactor', '◇', 'Propose a clearer visual draft of this part and its immediate connections. Explain tradeoffs. Return only supported diagram changes, not source edits.'],
    ['run', 'Run preview', '▷', 'Walk through this workflow hypothetically from the selected part. Label the answer as a simulation, distinguish unknowns, and do not execute anything or claim fresh results.'],
    ['explain', 'Explain', '▤', 'Explain this part’s responsibilities, inputs, outputs, and connections in plain language.'],
    ['worker', 'Generate worker', '✦', 'Design one additional visual-only worker stage beside this part, with a descriptive name and connection. Propose add_component and connect operations for review. Do not create executable code or run a worker.'],
  ];
  for (const [identifier, label, symbol, prompt] of prompts) {
    const button = action('', () => {
      closeNodeMenu();
      if (prompt === null) { state.tab = part.source ? 'source' : 'overview'; renderInspector(); }
      else askSelection('', prompt);
    });
    button.id = 'node-ai-' + identifier;
    button.append(element('span', 'menu-symbol', symbol), element('span', '', label));
    button.firstElementChild.setAttribute('aria-hidden', 'true');
    actions.append(button);
  }
  submenu.append(actions, element('p', 'node-menu-note', 'Review prompts before sending · drafts only'));
  submenu.hidden = false;
  const parent = select('#node-menu').getBoundingClientRect();
  const inspector = select('.inspector-panel').getBoundingClientRect();
  const availableRight = inspector.width ? inspector.left : innerWidth;
  const width = submenu.offsetWidth;
  const left = parent.right + width + 16 < availableRight ? parent.right + 10 : parent.left - width - 14;
  submenu.style.left = Math.max(12, Math.min(left, innerWidth - width - 12)) + 'px';
  submenu.style.top = Math.max(74, Math.min(parent.top + 50, innerHeight - submenu.offsetHeight - 42)) + 'px';
  state.tab = 'ai';
  renderInspector();
  select('#node-menu-ai').setAttribute('aria-expanded', 'true');
  select('#node-ai-ask').focus();
}

let menuComponent = null;

function closeNodeAIMenu(restoreFocus = false) {
  const submenu = select('#node-ai-menu');
  if (submenu) submenu.hidden = true;
  select('#node-menu-ai')?.setAttribute('aria-expanded', 'false');
  if (restoreFocus) select('#node-menu-ai')?.focus();
}

function closeNodeMenu(restoreFocus = false) {
  const menu = select('#node-menu');
  if (!menu) return;
  menu.hidden = true;
  closeNodeAIMenu();
  if (restoreFocus && menuComponent) select(`[data-node="${CSS.escape(menuComponent)}"]`)?.focus();
}

function positionNodeMenu(clientX, clientY) {
  const menu = select('#node-menu');
  const bounds = {width: menu.offsetWidth, height: menu.offsetHeight};
  const left = clientX + bounds.width + 24 < innerWidth ? clientX + 24 : clientX - bounds.width - 24;
  menu.style.left = Math.max(12, Math.min(left, innerWidth - bounds.width - 12)) + 'px';
  menu.style.top = Math.max(74, Math.min(clientY - 80, innerHeight - bounds.height - 42)) + 'px';
}

function openNodeMenu(identifier, clientX, clientY, keyboard = false) {
  const menu = select('#node-menu');
  if (!menu || state.preview || !component(identifier)) return;
  menuComponent = identifier;
  closeNodeAIMenu();
  select('#node-menu-title').textContent = component(identifier).name;
  select('#node-menu-detail').hidden = true;
  select('#node-menu-detail').replaceChildren();
  select('#node-menu-connect').disabled = scene().nodes.length < 2;
  select('#node-menu-disconnect').disabled = !scene().edges.some(edge => [edge.source, edge.target].includes(identifier));
  menu.hidden = false;
  positionNodeMenu(clientX, clientY);
  if (keyboard) menu.querySelector('button:not(:disabled)')?.focus();
}

function previewMenuAction(title, operations) {
  closeNodeMenu();
  try { previewProposal({title, description: 'Manual visual workspace edit. Harness source is unchanged.', operations}, state.sceneId); }
  catch (error) { toast(error.message); }
}

function showConnectionPicker(disconnect = false) {
  const source = menuComponent;
  const detail = select('#node-menu-detail');
  detail.replaceChildren();
  detail.hidden = false;
  const form = element('form', 'node-menu-picker');
  const label = element('label', '', disconnect ? 'Connection to remove' : 'Connect to');
  const picker = element('select');
  picker.id = 'node-menu-target';
  label.htmlFor = picker.id;
  const edges = scene().edges.filter(edge => [edge.source, edge.target].includes(source));
  const options = disconnect ? edges.map((edge, index) => ({value: String(index), label: `${component(edge.source).name} → ${component(edge.target).name}`})) : scene().nodes.filter(node => node.id !== source).map(node => ({value: node.id, label: component(node.id).name}));
  for (const option of options) {
    const item = element('option', '', option.label);
    item.value = option.value;
    picker.append(item);
  }
  form.append(label, picker);
  let connectionLabel;
  let connectionKind;
  if (!disconnect) {
    const nameLabel = element('label', '', 'Connection label');
    connectionLabel = element('input');
    connectionLabel.id = 'node-menu-edge-label';
    nameLabel.htmlFor = connectionLabel.id;
    connectionLabel.value = 'Proposed connection';
    connectionLabel.maxLength = 160;
    connectionLabel.required = true;
    const kindLabel = element('label', '', 'Connection type');
    connectionKind = element('select');
    connectionKind.id = 'node-menu-edge-kind';
    kindLabel.htmlFor = connectionKind.id;
    for (const kind of ['control', 'data', 'feedback']) {
      const option = element('option', '', human(kind));
      option.value = kind;
      connectionKind.append(option);
    }
    form.append(nameLabel, connectionLabel, kindLabel, connectionKind);
  }
  const submit = element('button', 'primary-button', 'Preview change');
  submit.type = 'submit';
  form.append(submit, element('small', '', 'Visual draft only · preview before applying'));
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (disconnect) {
      const edge = edges[Number(picker.value)];
      if (edge) previewMenuAction('Disconnect visual parts', [{type: 'disconnect', source: edge.source, target: edge.target}]);
    } else previewMenuAction('Connect visual parts', [{type: 'connect', source, target: picker.value, label: connectionLabel.value.trim(), kind: connectionKind.value}]);
  });
  detail.append(form);
  const menu = select('#node-menu');
  const bounds = menu.getBoundingClientRect();
  menu.style.top = Math.max(74, Math.min(bounds.top, innerHeight - bounds.height - 42)) + 'px';
  picker.focus();
}

function initializeNodeMenu() {
  const menu = select('#node-menu');
  if (!menu) return;
  select('#node-menu-close').addEventListener('click', () => closeNodeMenu(true));
  const submenu = element('section', 'node-menu node-ai-menu');
  submenu.id = 'node-ai-menu';
  submenu.hidden = true;
  submenu.setAttribute('role', 'dialog');
  submenu.setAttribute('aria-label', 'AI actions for selected part');
  document.body.append(submenu);
  select('#node-menu-ai').setAttribute('aria-haspopup', 'dialog');
  select('#node-menu-ai').setAttribute('aria-controls', 'node-ai-menu');
  select('#node-menu-ai').setAttribute('aria-expanded', 'false');
  select('#node-menu-ai').addEventListener('click', openNodeAIMenu);
  select('#node-menu-ai').addEventListener('keydown', event => { if (event.key === 'ArrowRight') { event.preventDefault(); openNodeAIMenu(); } });
  submenu.addEventListener('keydown', event => {
    if (['Escape', 'ArrowLeft'].includes(event.key)) { event.preventDefault(); event.stopPropagation(); closeNodeAIMenu(true); }
    if (!['ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const buttons = [...submenu.querySelectorAll('button')];
    const offset = event.key === 'ArrowDown' ? 1 : -1;
    buttons[(buttons.indexOf(document.activeElement) + offset + buttons.length) % buttons.length]?.focus();
  });
  select('#node-menu-connect').addEventListener('click', () => showConnectionPicker());
  select('#node-menu-disconnect').addEventListener('click', () => showConnectionPicker(true));
  select('#node-menu-duplicate').addEventListener('click', () => {
    const part = component(menuComponent);
    if (!part) return;
    const point = pointFor(menuComponent);
    const factor = state.exploded ? 1.45 : 1;
    previewMenuAction(`Duplicate ${part.name}`, [{type: 'add_component', component_id: 'draft-' + crypto.randomUUID(), name: short(part.name + ' copy', 80), description: part.description.slice(0, 1600), category: part.category, x: Math.max(0, Math.min(2400, point.x / factor + 208)), y: Math.max(0, Math.min(1600, point.y / factor + 104))}]);
  });
  select('#node-menu-delete').addEventListener('click', () => {
    const identifier = menuComponent;
    const part = component(identifier);
    if (!part) return;
    if (identifier.startsWith('draft-')) { previewMenuAction(`Delete ${part.name} from draft`, [{type: 'remove_component', component_id: identifier}]); return; }
    closeNodeMenu();
    const candidate = structuredClone(workspace());
    candidate.drafts[state.sceneId] ||= {nodes: [], edges: structuredClone(baseScene().edges), parameters: {}};
    const draft = candidate.drafts[state.sceneId];
    draft.hidden = [...new Set([...(draft.hidden || []), identifier])];
    draft.edges = draft.edges.filter(edge => edge.source !== identifier && edge.target !== identifier);
    try {
      validateWorkspace(candidate);
      showDraftPreview(candidate, state.sceneId, `Delete ${part.name} from visual draft`, [{type: 'hide_component', component_id: identifier}]);
    } catch (error) { toast(error.message); }
  });
  menu.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeNodeMenu(true); }
    if (!event.target.matches('button') || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const buttons = [...menu.querySelectorAll('button:not(:disabled)')].filter(button => button.offsetParent !== null);
    const current = buttons.indexOf(document.activeElement);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus();
  });
  document.addEventListener('pointerdown', event => { if (!event.target.closest('#node-menu, #node-ai-menu')) closeNodeMenu(); }, true);
  window.addEventListener('resize', () => closeNodeMenu());
  select('#assembly-svg').addEventListener('wheel', () => closeNodeMenu(), {passive: true});
  window.addEventListener('workbench:ask', () => closeNodeMenu());
}

function initialize() {
  loadPreferences();
  const aiTab = element('button', '', 'AI');
  aiTab.dataset.inspectorTab = 'ai';
  aiTab.type = 'button';
  aiTab.setAttribute('role', 'tab');
  aiTab.setAttribute('aria-selected', 'false');
  aiTab.setAttribute('aria-controls', 'inspector-content');
  select('.inspector-tabs').append(aiTab);
  select('#scene-select').value = state.sceneId;
  initializeCanvas();
  select('#component-search').addEventListener('input', () => { renderExplorer(); renderDiagram(); });
  select('#scene-select').addEventListener('change', (event) => switchScene(event.target.value));
  for (const button of selectAll('[data-view]')) button.addEventListener('click', () => switchView(button.dataset.view));
  for (const button of selectAll('[data-inspector-tab]')) button.addEventListener('click', () => { state.tab = button.dataset.inspectorTab; renderInspector(); });
  select('#inspector-close').setAttribute('aria-label', 'Hide component inspector');
  select('#inspector-close').title = 'Hide inspector; select a component to reopen';
  select('#inspector-close').addEventListener('click', () => { document.body.classList.remove('inspector-open'); document.body.classList.add('inspector-collapsed'); requestAnimationFrame(fitView); });
  select('#refresh-button').addEventListener('click', () => refresh());
  select('#save-workspace').addEventListener('click', () => { persist(); download('hyperspace-workspace.json', workspace()); toast('Workspace exported. Harness configuration is unchanged.'); });
  select('#load-workspace').addEventListener('click', () => select('#workspace-file').click());
  select('#workspace-file').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 200000) throw new Error('Workspace files must be smaller than 200 KB.');
      const value = JSON.parse(await file.text());
      validateWorkspace(value);
      const before = snapshotLayout();
      state.positions = value.positions;
      state.notes = value.notes || {};
      state.drafts = value.drafts || {};
      state.aiParts = value.ai_parts || {};
      switchScene(value.scene === 'research' ? 'research' : 'mission');
      remember(before);
      toast('Workspace layout and notes restored.');
    } catch (error) { toast(error.message); }
    event.target.value = '';
  });
  select('#explode-button').addEventListener('click', () => { const before = snapshotLayout(); state.exploded = !state.exploded; remember(before); renderDiagram(); updateToolbar(); fitView(); });
  select('#isolate-button').addEventListener('click', () => { state.isolated = !state.isolated; renderDiagram(); updateToolbar(); fitView(); });
  select('#reset-layout').addEventListener('click', () => { const before = snapshotLayout(); delete state.positions[state.sceneId]; state.exploded = false; state.isolated = false; remember(before); renderDiagram(); updateToolbar(); fitView(); toast('Assembly layout reset.'); });
  select('#undo-button').addEventListener('click', () => undo());
  select('#redo-button').addEventListener('click', () => undo(true));
  select('#zoom-in').addEventListener('click', () => zoom(1.2));
  select('#zoom-out').addEventListener('click', () => zoom(1 / 1.2));
  select('#fit-view').addEventListener('click', fitView);
  select('#run-select').addEventListener('change', (event) => loadRun(event.target.value));
  select('#play-button').addEventListener('click', togglePlayback);
  select('#timeline-slider').addEventListener('input', (event) => { stopPlayback(); setEvent(event.target.value); });
  select('#timeline-toggle').addEventListener('click', () => { const collapsed = select('#timeline-panel').classList.toggle('collapsed'); select('#timeline-toggle').setAttribute('aria-expanded', !collapsed); requestAnimationFrame(fitView); });
  select('#artifact-close').addEventListener('click', () => select('#artifact-dialog').close());
  select('#artifact-dialog').addEventListener('close', () => { state.artifactSequence += 1; });
  select('#artifact-dialog').addEventListener('click', (event) => { if (event.target === select('#artifact-dialog')) select('#artifact-dialog').close(); });
  select('#ask-component-button')?.addEventListener('click', () => askSelection(window.getSelection()?.toString()));
  select('#draft-apply')?.addEventListener('click', applyPreview);
  select('#draft-discard')?.addEventListener('click', () => dismissPreview());
  select('#draft-reset')?.addEventListener('click', () => {
    const before = snapshotLayout();
    for (const node of state.drafts[state.sceneId]?.nodes || []) {
      delete state.positions[state.sceneId]?.[node.id];
      delete state.notes[node.id];
      delete state.aiParts[node.id];
    }
    delete state.drafts[state.sceneId];
    for (const node of baseScene().nodes) delete state.aiParts[node.id];
    switchScene(state.sceneId);
    remember(before);
    toast('Saved architecture restored. Undo is available.');
  });
  select('#help-button')?.addEventListener('click', () => select('#help-dialog').showModal());
  select('#help-close')?.addEventListener('click', () => select('#help-dialog').close());
  document.addEventListener('contextmenu', (event) => {
    if (event.target.closest('#assistant-panel, #help-dialog, #artifact-dialog, input, textarea, select')) return;
    const part = event.target.closest('[data-node], [data-component]');
    const row = event.target.closest('[data-event-index]');
    if (!part && !row && !event.target.closest('#canvas-wrap, #inspector-content')) return;
    const selectedText = window.getSelection()?.toString() || '';
    event.preventDefault();
    if (part) {
      const identifier = part.dataset.node || part.dataset.component;
      if (!state.selection.has(identifier)) selectComponent(identifier);
    }
    if (row) { setEvent(row.dataset.eventIndex); const mapped = activeComponent(); if (mapped) selectComponent(mapped); }
    askSelection(selectedText);
  });
  document.addEventListener('keydown', (event) => {
    if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || select('#artifact-dialog').open || select('#help-dialog')?.open || event.target.closest('#assistant-panel, #node-menu, #node-ai-menu')) return;
    if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) { event.preventDefault(); askSelection(); return; }
    if (event.key === '/') { event.preventDefault(); select('#component-search').focus(); return; }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); undo(event.shiftKey); return; }
    if (state.view !== 'assembly') return;
    if (event.key === 'Escape') { dismissPreview(); state.isolated = false; document.body.classList.remove('inspector-open'); renderDiagram(); updateToolbar(); fitView(); }
    if (event.key.toLowerCase() === 'f') fitView();
    if (event.key.toLowerCase() === 'e') select('#explode-button').click();
    if (event.key.toLowerCase() === 'i') select('#isolate-button').click();
    if (state.selected && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
      event.preventDefault();
      if (state.preview) { toast('Apply or dismiss the preview before moving parts.'); return; }
      const before = snapshotLayout();
      const point = pointFor(state.selected);
      const step = event.shiftKey ? 64 : 16;
      if (event.key === 'ArrowUp') point.y -= step;
      if (event.key === 'ArrowDown') point.y += step;
      if (event.key === 'ArrowLeft') point.x -= step;
      if (event.key === 'ArrowRight') point.x += step;
      setPoint(state.selected, point);
      remember(before); renderDiagram();
    }
  });
  let resizeTimer;
  new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(fitView, 100); }).observe(select('#canvas-wrap'));
  try { changePreferences(JSON.parse(localStorage.getItem(PREFERENCES_KEY) || '{}'), false); }
  catch { changePreferences(DEFAULT_PREFERENCES, false); }
  select('#live-toggle').addEventListener('change', event => changePreferences({auto_refresh: event.target.checked}));
  updateToolbar();
  state.selection = new Set([state.sceneId === 'research' ? 'research-plan' : 'coordinator']);
  state.selected = [...state.selection][0];
  window.hyperspaceWorkbench = {context: assistantContext, preview: previewProposal, dismissPreview, applyPreview, ask: askSelection, hasPreview: () => Boolean(state.preview), assistantBusy: () => Boolean(window.hyperspaceAssistant?.busy()), settings: () => switchView('settings')};
  window.hyperspacePreferences = {get: () => ({...preferences}), set: changePreferences, reset: () => changePreferences(DEFAULT_PREFERENCES)};
  initializeNodeMenu();
  refresh(true);
}

initialize();
