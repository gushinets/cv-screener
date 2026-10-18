import test from 'node:test';
import assert from 'node:assert/strict';
import {api, post, sourceAt, citationPlugin} from './api.js';

test('uploads keep the browser multipart boundary while chat requests remain JSON', async () => {
  const previous = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => { calls.push({url, ...options}); return {ok: true, json: async () => ({status: 'queued'})}; };
  try {
    const body = new FormData(); body.append('file', new Blob(['Resume']), 'Candidate.txt');
    await api('/resume-imports', {method: 'POST', body});
    await post('/chat', {message: 'Find a candidate'});
    assert.equal(calls[0].body, body);
    assert.equal(calls[0].headers['Content-Type'], undefined);
    assert.equal(calls[1].headers['Content-Type'], 'application/json');
    assert.equal(JSON.parse(calls[1].body).message, 'Find a candidate');
  } finally { globalThis.fetch = previous; }
});

test('source numbers are resolved within their own reply snapshot', () => {
  const old = {sources: [{name: 'Emily.txt', text: 'old snapshot'}]};
  const current = {sources: [{name: 'Bob.txt', text: 'new snapshot'}, ...old.sources]};
  assert.equal(sourceAt(old, 1).name, 'Emily.txt');
  assert.equal(sourceAt(current, 1).name, 'Bob.txt');
  assert.equal(sourceAt(current, 2).text, 'old snapshot');
  assert.equal(sourceAt(old, 2), undefined);
  assert.equal(sourceAt(old, 0), undefined);
});

test('citation transformer preserves prose and leaves code and existing links unchanged', () => {
  const tree = {type: 'root', children: [{type: 'paragraph', children: [
    {type: 'text', value: 'Emily [1], Bob [2].'}, {type: 'inlineCode', value: '[99]'},
    {type: 'link', url: 'https://example.com', children: [{type: 'text', value: '[8]'}]},
  ]}]};
  citationPlugin()(tree);
  const children = tree.children[0].children;
  assert.deepEqual(children.filter(n => n.type === 'link').map(n => n.url), ['#cv-source-1', '#cv-source-2', 'https://example.com']);
  assert.equal(children.find(n => n.type === 'inlineCode').value, '[99]');
  assert.equal(children.map(n => n.value || n.children[0].value).join(''), 'Emily [1], Bob [2].[99][8]');
});
