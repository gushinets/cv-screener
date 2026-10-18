// Run with agent-browser eval -b in an isolated session on the local application.
(async () => {
  const assert = (condition, message) => { if (!condition) throw Error(message); };
  const waitFor = async check => {
    const deadline = performance.now() + 10000;
    while (!check()) {
      assert(performance.now() < deadline, 'Timed out waiting for the interface');
      await new Promise(requestAnimationFrame);
    }
  };
  const read = async path => {
    const response = await fetch('/api' + path);
    assert(response.ok, 'Local API unavailable');
    return response.json();
  };
  const id = new URLSearchParams(location.search).get('id');
  assert(id, 'Open a saved conversation before running this check');
  const original = await read('/conversations/' + id);
  const favorites = await read('/favorites');
  const replies = original.messages.filter(m => m.role === 'assistant' && m.sources?.length);
  assert(replies.length, 'This check needs an answer with saved sources');
  const sidebar = document.querySelector('.sidebar');
  const links = [...document.querySelectorAll('.view-tabs a')];
  assert(links.length === 4, 'Expected four conversation views');
  links[0].click();
  await waitFor(() => document.querySelector('#query'));
  const accent = getComputedStyle(document.body).getPropertyValue('--accent');
  document.querySelector('.view-tabs a:last-child').click();
  await waitFor(() => document.querySelector('#reply-select'));
  assert(document.querySelector('.sidebar') === sidebar, 'Navigation was replaced when opening the report');
  assert(getComputedStyle(document.body).getPropertyValue('--accent') === accent, 'Report changed the palette');
  assert(document.querySelectorAll('h1').length === 1, 'Report must share the page heading');
  let checked = 0;
  for (const reply of replies) {
    const select = document.querySelector('#reply-select');
    select.value = reply.id;
    select.dispatchEvent(new Event('change', {bubbles: true}));
    await waitFor(() => document.querySelector('a[download]')?.getAttribute('href').endsWith('/' + reply.id));
    const citation = document.querySelector('.recommendation .cite,.evidence-link');
    assert(citation, 'Saved answer has no accessible source link');
    const number = Number(citation.textContent.match(/\[(\d+)\]/)[1]);
    citation.focus(); citation.click();
    await waitFor(() => document.querySelector('#source-dialog').open);
    assert(document.querySelector('.source-full').textContent === reply.sources[number - 1].text, 'Source belongs to a different answer');
    document.querySelector('#source-dialog .icon-button').click();
    await waitFor(() => !document.querySelector('#source-dialog').open && !document.querySelector('#source-title'));
    const response = await fetch(document.querySelector('a[download]').href);
    assert(response.ok && response.headers.get('Content-Disposition')?.includes('attachment'), 'Report is not downloadable');
    const markdown = await response.text();
    assert(reply.sources.every(source => markdown.includes(source.text)), 'Export lost saved source text');
    checked++;
  }
  assert(JSON.stringify(await read('/conversations/' + id)) === JSON.stringify(original), 'Conversation changed during navigation');
  assert(JSON.stringify(await read('/favorites')) === JSON.stringify(favorites), 'Favorites changed during navigation');
  assert(document.documentElement.scrollWidth <= innerWidth, 'Page overflows horizontally');
  return {checkedAnswers: checked, historyUnchanged: true, favoritesUnchanged: true, sharedNavigation: true, sourceSnapshotsExact: true};
})()
