import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { createGateway, GatewayJournal, GATEWAY_VERSION } from '../photon_gateway.mjs';

const HOME = 'test-home-space';
const OPERATOR = '+15555550101';
const space = { id: HOME };
function message(id = 'test-provider-id', overrides = {}) {
  return { id, direction: 'inbound', sender: { id: OPERATOR }, timestamp: new Date('2026-09-08T01:02:03Z'),
    content: { type: 'text', text: 'A test message' }, ...overrides };
}
function setup(t, options = {}) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'photon-gateway-test-'));
  const sent = [];
  const gateway = createGateway({ directory, homeSpace: HOME, operatorNumber: OPERATOR,
    sendText: async (recipient, text) => { sent.push({ recipient, text }); return { id: 'outbound-test-id' }; }, ...options });
  t.after(async () => { await gateway.close(); fs.rmSync(directory, { recursive: true, force: true }); });
  return { gateway, directory, sent };
}
function request(base, method, endpoint, body, headers = {}) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? undefined : JSON.stringify(body);
    const req = http.request(new URL(endpoint, base), { method, headers: {
      ...(payload ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) } : {}), ...headers } }, res => {
      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(Buffer.concat(chunks).toString()) }));
    });
    req.on('error', reject);
    req.end(payload);
  });
}

test('only exact operator, home space, explicit inbound, and stable IDs enter journal', async t => {
  const { gateway } = setup(t);
  for (const [where, value] of [
    [{ id: 'other-space' }, message()], [space, message('a', { sender: { id: '+15555550102' } })],
    [space, message('a', { direction: 'outbound' })], [space, message('a', { direction: undefined })],
    [space, message('', {})], [space, message('a', { content: { type: 'reaction', emoji: 'heart' } })],
  ]) assert.equal(gateway.accept(where, value).accepted, false);
  assert.equal(gateway.journal.messages.length, 0);
  assert.equal(gateway.accept(space, message()).accepted, true);
  assert.equal(gateway.journal.messages.length, 1);
});

test('provider ID dedup survives restart and API omits sender, recipient, and native ID', async t => {
  const { gateway, directory } = setup(t);
  const first = gateway.accept(space, message());
  const journalId = gateway.journal.inbox().journal_id;
  assert.equal(typeof journalId, 'string');
  assert.equal(gateway.accept(space, message()).reason, 'duplicate');
  await gateway.close();
  const resumed = createGateway({ directory, homeSpace: HOME, operatorNumber: OPERATOR, sendText: async () => {} });
  try {
    assert.equal(resumed.accept(space, message()).reason, 'duplicate');
    const inbox = resumed.journal.inbox();
    assert.equal(inbox.journal_id, journalId);
    assert.equal(inbox.messages[0].event_id, first.message.event_id);
    assert.equal(inbox.next_cursor, 1);
    assert.equal(inbox.latest_seq, 1);
    for (const field of ['sender', 'space_id', 'source_message_id']) assert.equal(field in inbox.messages[0], false);
    assert.equal(JSON.stringify(inbox).includes(OPERATOR), false);
    assert.equal(fs.statSync(path.join(directory, 'inbound.jsonl')).mode & 0o777, 0o600);
  } finally { await resumed.close(); }
});

test('intake is fsynced before visibility; partial final frame recovers without deleting committed history', async t => {
  const { gateway, directory } = setup(t);
  gateway.accept(space, message());
  const disk = fs.readFileSync(path.join(directory, 'inbound.jsonl'), 'utf8');
  assert.equal(JSON.parse(disk.trim().split('\n').at(-1)).record.seq, 1);
  const namespace = gateway.journal.namespace;
  await gateway.close();
  fs.appendFileSync(path.join(directory, 'inbound.jsonl'), '{"record":');
  const journal = new GatewayJournal(directory, namespace);
  try { assert.equal(journal.inbox().latest_seq, 1); assert.equal(fs.readFileSync(journal.file, 'utf8'), disk); }
  finally { journal.close(); }
});

test('committed checksum corruption and conversation binding changes fail closed', async t => {
  const { gateway, directory } = setup(t);
  gateway.accept(space, message());
  const namespace = gateway.journal.namespace;
  await gateway.close();
  assert.throws(() => new GatewayJournal(directory, 'different-binding'), /binding/);
  fs.appendFileSync(path.join(directory, 'inbound.jsonl'), '{"record":{"kind":"sent","source_message_id":"x"},"sha256":"bad"}\n');
  assert.throws(() => new GatewayJournal(directory, namespace), /checksum/);
});

test('cursor replay is non-draining and validates bounds', async t => {
  const { gateway } = setup(t);
  gateway.accept(space, message('one'));
  gateway.accept(space, message('two'));
  const first = gateway.journal.inbox(0, 1);
  assert.equal(first.messages[0].seq, 1);
  assert.deepEqual(gateway.journal.inbox(0, 1), first);
  assert.equal(gateway.journal.inbox(first.next_cursor, 1).messages[0].seq, 2);
  assert.deepEqual(gateway.journal.inbox(2).messages, []);
  for (const cursor of [-1, 3, 0.1, NaN]) assert.throws(() => gateway.journal.inbox(cursor), /cursor/);
});

test('bounded journal capacity blocks intake explicitly', async t => {
  const { gateway } = setup(t, { journalLimits: { maxEntries: 1 } });
  gateway.accept(space, message('one'));
  assert.throws(() => gateway.accept(space, message('two')), /journal_capacity/);
  assert.equal(gateway.health().blocked, 'journal_capacity');
  assert.equal(gateway.health().latest_seq, 1);
  assert.equal(gateway.accept(space, message('three')).reason, 'blocked');
});

test('unexpected journal I/O failure blocks intake instead of reporting a malformed message', async t => {
  const { gateway } = setup(t);
  gateway.journal.append = () => { throw new Error('injected_local_io_failure'); };
  assert.throws(() => gateway.accept(space, message()), /journal_failure/);
  assert.equal(gateway.health().blocked, 'journal_failure');
  assert.equal(gateway.journal.messages.length, 0);
});

test('groups preserve bounded text and attachment metadata without loading attachment bytes', async t => {
  const { gateway } = setup(t);
  const result = gateway.accept(space, message('group', { content: { type: 'group', items: [
    { content: { type: 'text', text: 'One' } }, { content: { type: 'markdown', markdown: 'Two' } },
    { content: { type: 'attachment', name: 'test.png', mimeType: 'image/png', data: () => { throw new Error('must not execute'); } } },
  ] } }));
  assert.equal(result.message.text, 'One\nTwo');
  assert.deepEqual(result.message.attachment, { filename: 'test.png', mime: 'image/png' });
  assert.equal(gateway.accept(space, message('huge', { content: { type: 'text', text: 'x'.repeat(16001) } })).accepted, false);
});

test('loopback HTTP exposes durable inbox and text-only fixed-recipient send', async t => {
  const { gateway, sent } = setup(t);
  const base = await gateway.listen(0);
  assert.equal((await request(base, 'GET', '/health')).body.version, GATEWAY_VERSION);
  assert.equal((await request(base, 'POST', '/send', { text: 'Reply' })).status, 503);
  gateway.setConnected(true);
  assert.equal((await request(base, 'POST', '/send', { text: 'Reply', space_id: 'different' })).status, 403);
  assert.equal((await request(base, 'POST', '/send', { text: 'Reply', image: '/tmp/no-file-access' })).status, 400);
  assert.equal((await request(base, 'POST', '/send', { text: 'Reply' }, { Origin: 'https://untrusted.example' })).status, 403);
  assert.equal((await request(base, 'POST', '/send', { text: 'Reply' }, { 'Content-Type': 'text/plain' })).status, 400);
  assert.equal((await request(base, 'POST', '/send', { text: 'Reply' })).status, 200);
  assert.deepEqual(sent, [{ recipient: HOME, text: 'Reply' }]);
  assert.equal(gateway.accept(space, message('outbound-test-id')).reason, 'self_echo');
  gateway.accept(space, message('incoming'));
  const inbox = await request(base, 'GET', '/inbox?after=0&limit=1');
  assert.equal(inbox.body.messages[0].seq, 1);
  assert.equal((await request(base, 'GET', '/inbox?after=999')).status, 400);
  assert.equal((await request(base, 'GET', '/inbound')).status, 410);
});

test('ambiguous provider send exposes no raw provider error or credential-like detail', async t => {
  const { gateway } = setup(t, { sendText: async () => { throw new Error('private-provider-detail'); } });
  const base = await gateway.listen(0);
  gateway.setConnected(true);
  const response = await request(base, 'POST', '/send', { text: 'Reply' });
  assert.equal(response.status, 502);
  assert.deepEqual(response.body, { error: 'send_outcome_unknown' });
  assert.equal(JSON.stringify(gateway.health()).includes('private-provider-detail'), false);
});
