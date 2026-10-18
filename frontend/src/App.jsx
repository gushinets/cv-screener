import React, {useEffect, useRef, useState} from 'react';
import Icon from './Icons.jsx';
import MessageMarkdown, {Citation} from './Markdown.jsx';
import {api, post, sourceAt} from './api.js';
import ResumeLibrary from './ResumeLibrary.jsx';

const NavigationContext = React.createContext(null);
function Nav({changes, children, ...props}) {
  const {href, navigate} = React.useContext(NavigationContext);
  return <a href={href(changes)} {...props} onClick={event => {
    if (!event.metaKey && !event.ctrlKey && !event.shiftKey && event.button === 0) {event.preventDefault(); navigate(changes);}
  }}>{children}</a>;
}

function currentRoute() {
  const params = new URLSearchParams(location.search);
  return {id: params.get('id'), turn: params.get('turn'),
    view: ['candidates', 'chat', 'compare', 'report', 'saved', 'library'].includes(params.get('view')) ? params.get('view') : 'candidates'};
}
const initials = name => name.split(/\s+/).slice(0, 2).map(s => s[0]).join('').toUpperCase();
const date = value => value ? new Intl.DateTimeFormat('ru', {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'}).format(new Date(value)) : 'Дата не сохранена';

function citedSources(message) {
  if (!message) return [];
  const numbers = new Set([...message.content.matchAll(/\[(\d+)\]/g)].map(m => Number(m[1])));
  message.analysis?.candidates.forEach(c => numbers.add(c.source_id));
  return (message.sources || []).map((source, index) => ({source, number: index + 1}))
    .filter(entry => message.analysis?.candidates.length === 0 || !numbers.size || numbers.has(entry.number));
}

function Matrix({message, onSource}) {
  const result = message.analysis;
  if (!result?.criteria.length || !result.candidates.length) return <MessageMarkdown message={message} onSource={onSource} />;
  return <><p className="matrix-hint">Прокрутите таблицу вправо, чтобы увидеть всех кандидатов.</p>
    <div className="matrix-scroll" role="region" tabIndex={0} aria-label="Сравнение кандидатов">
      <table className="matrix"><caption className="sr-only">Критерии из текущего ответа</caption>
        <thead><tr><th scope="col">Критерий</th>{result.candidates.map(c => <th scope="col" key={c.source_id}>{c.name}<small>{c.role}</small></th>)}</tr></thead>
        <tbody>{result.criteria.map((criterion, index) => <tr key={index}><th scope="row">{criterion.title}</th>
          {result.candidates.map(c => {
            const cell = criterion.cells.find(item => item.source_id === c.source_id);
            return <td key={c.source_id}>{cell && <><span className={`cell-status ${cell.status === 'confirmed' ? '' : 'partial'}`}><Icon name={cell.status === 'confirmed' ? 'file' : 'dash'} />{cell.status === 'confirmed' ? 'По резюме' : 'Уточнить'}</span><div className="fact-copy"><MessageMarkdown message={message} text={cell.text} onSource={onSource} /> {!cell.text.includes(`[${c.source_id}]`) && <Citation message={message} number={c.source_id} onSource={onSource} />}</div></>}</td>;
          })}</tr>)}</tbody>
      </table>
    </div></>;
}

function SourceDialog({selection, onClose}) {
  const dialog = useRef(null);
  const [tab, setTab] = useState('snapshot');
  const [full, setFull] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!selection) return;
    setTab('snapshot'); setFull(null); setError('');
    dialog.current.showModal(); document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; if (dialog.current?.open) dialog.current.close(); };
  }, [selection]);
  useEffect(() => {
    if (!selection || tab !== 'full') return;
    let ignore = false; setError('');
    api(`/resume?name=${encodeURIComponent(selection.source.name)}`)
      .then(value => { if (!ignore) setFull(value.text); }).catch(err => { if (!ignore) setError(err.message); });
    return () => { ignore = true; };
  }, [selection, tab]);
  return <dialog ref={dialog} id="source-dialog" aria-labelledby="source-title" onClose={onClose}>
    {selection && <><header className="source-head"><div><small>{selection.number ? `Источник [${selection.number}] этого ответа` : 'Резюме из базы'}</small>
      <h2 id="source-title">{selection.source.name}</h2><p>Проверка первоисточника</p></div>
      <button className="icon-button" onClick={() => dialog.current.close()} aria-label="Закрыть источник"><Icon name="close" /></button></header>
      {selection.number && <div className="source-tabs"><button aria-pressed={tab === 'snapshot'} onClick={() => setTab('snapshot')}>Сохранённые фрагменты</button>
        <button aria-pressed={tab === 'full'} onClick={() => setTab('full')}>Файл резюме</button></div>}
      <div className="source-document"><p>{!selection.number ? 'Текст текущего файла резюме из базы.' : tab === 'snapshot' ? 'Эти фрагменты сохранены вместе с выбранным ответом. Их номера не меняются при новых запросах.' : 'Текущая версия файла из базы. Сохранённые фрагменты предыдущих ответов остаются неизменными.'}</p>
        {error && tab === 'full' ? <p className="error-box" role="alert">{error}</p> : <div className="source-full">{tab === 'snapshot' ? selection.source.text : full ?? 'Загружаем резюме…'}</div>}
      </div></>}
  </dialog>;
}

export default function App() {
  const [route, setRoute] = useState(currentRoute);
  const routeRef = useRef(route); routeRef.current = route;
  const epoch = useRef(0);
  const [boot, setBoot] = useState(null);
  const [threads, setThreads] = useState([]);
  const [cache, setCache] = useState({});
  const [favorites, setFavorites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [conversationError, setConversationError] = useState(null);
  const [conversationAttempt, setConversationAttempt] = useState(0);
  const [busy, setBusy] = useState(null);
  const [toast, setToast] = useState('');
  const [selection, setSelection] = useState(null);
  const [drafts, setDrafts] = useState(() => { try { const saved = JSON.parse(sessionStorage.getItem('cv-screener-drafts-v1') || '{}'); return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {}; } catch { return {}; } });
  const retry = useRef(null);
  const input = useRef(null);
  const conversation = route.id ? cache[route.id] : null;
  const replies = conversation?.messages.filter(m => m.role === 'assistant') || [];
  const reply = replies.find(m => m.id === route.turn) || replies.at(-1);
  const isReport = route.view === 'report';
  const draftKey = route.id || 'new';
  const draft = typeof drafts[draftKey] === 'string' ? drafts[draftKey] : '';

  function href(changes = {}) {
    const values = {...route, ...changes};
    return `/?${new URLSearchParams(Object.entries(values).filter(([, value]) => value))}`;
  }
  function navigate(changes, replace = false) {
    const next = {...routeRef.current, ...changes}; epoch.current += 1;
    const target = `/?${new URLSearchParams(Object.entries(next).filter(([, value]) => value))}`;
    history[replace ? 'replaceState' : 'pushState']({}, '', target);
    routeRef.current = next; setRoute(next); setError('');
    window.scrollTo(0, 0);
  }
  function setDraft(value) { setDrafts(old => ({...old, [draftKey]: value})); }
  async function refreshThreads() { const value = await api('/conversations'); setThreads(value); return value; }

  useEffect(() => {
    let ignore = false;
    (async () => {
      try {
        const [settings, saved] = await Promise.all([api('/bootstrap'), api('/favorites')]);
        if (ignore) return; setBoot(settings); setFavorites(saved);
        const list = await refreshThreads();
        if (!ignore && !routeRef.current.id && !location.search && list.length) navigate({id: list[0].id}, true);
      } catch (err) { if (!ignore) setError(err.message); }
      finally { if (!ignore) setLoading(false); }
    })();
    const pop = () => { epoch.current += 1; setRoute(currentRoute()); };
    addEventListener('popstate', pop);
    return () => { ignore = true; removeEventListener('popstate', pop); };
  }, []);
  useEffect(() => {
    try { sessionStorage.setItem('cv-screener-drafts-v1', JSON.stringify(drafts)); } catch { /* Draft remains in this tab. */ }
  }, [drafts]);
  useEffect(() => {
    document.body.className = isReport ? 'report' : '';
    document.title = `${conversation?.title || 'Подбор кандидатов'} | CV Screener`;
  }, [isReport, conversation?.title]);
  useEffect(() => {
    if (!route.id || cache[route.id]) return;
    let ignore = false; setConversationError(null);
    api(`/conversations/${encodeURIComponent(route.id)}`).then(value => {
      if (!ignore) setCache(old => ({...old, [value.id]: value}));
    }).catch(err => { if (!ignore) setConversationError({id: route.id, message: err.message}); });
    return () => { ignore = true; };
  }, [route.id, cache, conversationAttempt]);
  useEffect(() => { if (!toast) return; const timer = setTimeout(() => setToast(''), 3200); return () => clearTimeout(timer); }, [toast]);

  async function send(event) {
    event.preventDefault();
    if (busy) return;
    const message = draft.trim();
    if (!message) { setError('Опишите вакансию или задайте вопрос о кандидатах.'); input.current?.focus(); return; }
    const id = route.id, revision = conversation?.revision || 0, startedEpoch = epoch.current;
    let body = retry.current;
    if (!body || body.message !== message || body.conversation_id !== id || body.revision !== revision)
      body = {conversation_id: id, revision, message, request_id: crypto.randomUUID()};
    retry.current = body; setBusy({id, message}); setError('');
    try {
      const result = await post('/chat', body);
      setCache(old => ({...old, [result.id]: result}));
      await refreshThreads(); retry.current = null;
      setDrafts(old => old[id || 'new']?.trim() === message ? {...old, [id || 'new']: ''} : old);
      if (epoch.current === startedEpoch) navigate({id: result.id, turn: null, view: 'candidates'});
      else setToast('Ответ готов и сохранён в истории подборов');
    } catch (err) {
      setError(err.message);
      if (err.status === 409 && id) {
        api(`/conversations/${id}`).then(value => setCache(old => ({...old, [id]: value}))).catch(() => {});
      }
    } finally { setBusy(null); }
  }

  async function toggleFavorite(candidate, message, conversationId = conversation?.id) {
    const name = sourceAt(message, candidate.source_id)?.name;
    if (!name) return;
    try {
      const exists = favorites.some(f => f.name === name);
      const result = exists ? await api(`/favorites/${encodeURIComponent(name)}`, {method: 'DELETE'})
        : await post('/favorites', {conversation_id: conversationId, message_id: message.id, source_id: candidate.source_id});
      setFavorites(result); setToast(exists ? 'Кандидат убран из избранного' : 'Кандидат добавлен в избранное');
    } catch (err) { setError(err.message); }
  }
  function favoriteButton(candidate, message, cid) {
    const saved = favorites.some(f => f.name === sourceAt(message, candidate.source_id)?.name);
    return <button className="icon-button" onClick={() => toggleFavorite(candidate, message, cid)} aria-pressed={saved}
      aria-label={`${saved ? 'Убрать из избранного' : 'В избранное'}: ${candidate.name}`}><Icon name="star" /></button>;
  }

  function renderCard({candidate, message, cid}) {
    const recommended = message.analysis?.recommended_source_id === candidate.source_id;
    return <article key={candidate.source_id} className={`candidate ${recommended ? 'recommended' : ''}`}>
      <div className="candidate-body"><div className="candidate-head"><div className="avatar" aria-hidden="true">{initials(candidate.name)}</div>
        <div className="candidate-title"><div className="candidate-name"><h2>{candidate.name}</h2>{recommended && <span className="recommendation-tag">Рекомендуем</span>}</div><p>{candidate.role}</p></div>{favoriteButton(candidate, message, cid)}</div>
        <div className="candidate-evidence"><div className="candidate-note"><MessageMarkdown message={message} text={candidate.summary} onSource={setSelection} /></div>
        {candidate.facts.map((fact, i) => <div className="fact-row" key={i}><Icon name="check" /><div className="fact-copy"><MessageMarkdown message={message} text={fact.text} onSource={setSelection} /> {!fact.text.includes(`[${candidate.source_id}]`) && <Citation message={message} number={candidate.source_id} onSource={setSelection} />}</div></div>)}
        {candidate.gaps.length > 0 && <div className="limit-note"><strong>На интервью: </strong><MessageMarkdown message={message} text={candidate.gaps.join(' ')} onSource={setSelection} /></div>}
        <div className="candidate-bottom"><button className="text-button" onClick={() => setSelection({source: sourceAt(message, candidate.source_id), number: candidate.source_id, messageId: message.id})}><Icon name="file" /> Открыть резюме</button></div>
      </div></div></article>;
  }

  function renderQuery() {
    return <form className="query-box live-query" onSubmit={send}>
      <label htmlFor="query">{conversation ? 'Уточнить подбор' : 'Кого ищем'}</label>
      <div className="query-row"><textarea ref={input} id="query" name="query" rows={2} maxLength={12000} autoComplete="off" value={draft}
        onChange={event => setDraft(event.target.value)} placeholder="Например, DevOps с Kubernetes, Terraform и AWS…" aria-describedby={error ? 'request-error' : undefined} />
        <button className="button primary" type="submit" disabled={!!busy || (route.id && !conversation)}><Icon name={busy ? 'info' : 'search'} />{busy ? 'Готовим ответ…' : conversation ? 'Спросить' : 'Подобрать'}</button></div>
      {busy && <p className="generation-status" role="status">Проверяем резюме и формируем ответ. Результат сохранится в истории.</p>}
      <details className="query-examples" open={!conversation}><summary>Примеры запросов</summary><div className="query-footer">{boot.examples.map((example, index) => <button className="example" type="button" key={example} onClick={() => {setDraft(example); input.current?.focus();}}>{['Найти DevOps', 'Сравнить Python', 'Вопросы для ML'][index]}</button>)}</div></details>
    </form>;
  }

  function renderChatHistory() {
    return <section className="chat-transcript" aria-label="Сообщения диалога">{conversation.messages.map(message => <article className={`chat-message ${message.role}`} key={message.id}>
      <div className="message-heading"><h2>{message.role === 'user' ? 'Ваш запрос' : 'Ответ по резюме'}</h2><span>{date(message.created_at)}</span></div>
      {message.role === 'assistant' ? <><MessageMarkdown message={message} onSource={setSelection} />
        <div className="message-actions"><Nav changes={{view: 'report', turn: message.id}} className="text-button">Открыть отчёт этого ответа <Icon name="file" /></Nav><span>{message.sources?.length || 0} сохранённых источников</span></div></> : <p className="user-query">{message.content}</p>}
    </article>)}</section>;
  }

  function renderConversationState() {
    if (conversationError?.id !== route.id) return <p className="loading">Загружаем диалог…</p>;
    return <section className="empty"><h2>Не удалось открыть диалог</h2><p role="alert">{conversationError.message}</p>
      <button className="button" onClick={() => setConversationAttempt(value => value + 1)}>Повторить загрузку</button>{' '}
      <Nav changes={{id: null, turn: null, view: 'candidates'}} className="button">Новый подбор</Nav></section>;
  }

  function renderResults() {
    if (viewIs('library')) return <ResumeLibrary library={boot.library} onRefresh={setBoot} onSource={setSelection} />;
    if (viewIs('saved')) return favorites.length ? <section className="candidate-list" aria-label="Избранные кандидаты">{favorites.map(f => {
      const message = {id: f.message_id, sources: Array.from({length: f.candidate.source_id}, (_, i) => i === f.candidate.source_id - 1 ? f.source : null), analysis: null};
      return <div key={f.name}>{renderCard({candidate:f.candidate, message, cid:f.conversation_id})}<Nav changes={{id: f.conversation_id, turn: f.message_id, view: 'chat'}} className="text-button">Перейти к исходному диалогу</Nav></div>;
    })}</section> : <section className="empty"><h2>Сохраните кандидата для обсуждения</h2><p>Нажмите звезду в карточке. Избранное хранится вместе с историей на этом компьютере.</p><Nav changes={{view: 'candidates'}} className="button">К подбору</Nav></section>;
    if (!route.id && !reply) return <section className="welcome-panel"><h2>Что указать в запросе</h2><p>Добавьте роль, обязательные навыки и задачи будущего сотрудника. После первого ответа можно уточнить требования, сравнить кандидатов или открыть отчёт.</p></section>;
    if (!conversation) return renderConversationState();
    return <>{viewIs('chat') ? renderChatHistory() : reply && <section aria-label="Результаты подбора">
        {viewIs('compare') ? <Matrix message={reply} onSource={setSelection} /> : <>
          <div className="answer-summary"><h2>Вывод</h2><MessageMarkdown message={reply} onSource={setSelection} /></div>
          <div className="candidate-list">{reply.analysis?.candidates.map(c => renderCard({candidate:c,message:reply}))}</div></>}
      </section>}
    </>;
  }
  function viewIs(name) { return route.view === name; }

  function renderReport() {
    const result = reply?.analysis;
    if (route.id && !conversation) return renderConversationState();
    if (!reply) return <section className="empty"><h2>Для отчёта нужен ответ</h2><p>Вернитесь к подбору и задайте вопрос о кандидатах.</p><Nav changes={{view: 'candidates'}} className="button">К подбору</Nav></section>;
    return <><div className="report-toolbar"><div className="reply-control"><label htmlFor="reply-select">Ответ для отчёта</label><select id="reply-select" name="reply" value={reply.id} onChange={event => navigate({turn: event.target.value})}>{replies.map((m, i) => <option key={m.id} value={m.id}>Ответ {i + 1}: {date(m.created_at)}</option>)}</select></div></div>
        <div className="report-layout"><article className="report-sheet"><header className="report-heading"><div className="document-meta"><span className="doc-type"><Icon name="file" /> Отчёт по подбору</span><span>{date(reply.created_at)}</span></div><p>Ответ на запрос: {conversation.messages.slice(0, conversation.messages.findIndex(m => m.id === reply.id)).findLast(m => m.role === 'user')?.content}</p></header>
          <section className="recommendation"><h2>Вывод</h2><MessageMarkdown message={reply} text={result?.recommendation || reply.content} onSource={setSelection} /></section>
          {result?.candidates.length > 0 && <><section><div className="section-heading"><h2>Сравнение по требованиям</h2><small>Ссылки ведут к источникам этого ответа</small></div><Matrix message={reply} onSource={setSelection} /></section>
            <section className="report-candidates"><h2>Подтверждённые факты</h2>{result.candidates.map(c => <section key={c.source_id}><h3>{c.name} <Citation message={reply} number={c.source_id} onSource={setSelection} /></h3><ul>{c.facts.map((f, i) => <li className="fact-copy" key={i}><MessageMarkdown message={reply} text={f.text} onSource={setSelection} /> {!f.text.includes(`[${c.source_id}]`) && <Citation message={reply} number={c.source_id} onSource={setSelection} />}</li>)}</ul>{c.gaps.length > 0 && <div className="muted fact-copy">На интервью: <MessageMarkdown message={reply} text={c.gaps.join(' ')} onSource={setSelection} /></div>}</section>)}</section></>}
          {!result?.candidates.length && result && <section><h2>Основания ответа</h2><MessageMarkdown message={reply} onSource={setSelection} /></section>}
          <section className="report-limits"><h2>Границы вывода</h2><p>Оцениваем сведения в резюме. Отсутствие упоминания не доказывает отсутствие навыка. Практический уровень нужно проверить на интервью.</p></section>
          <footer className="report-footer"><span>CV Screener</span><span>Источники сохранены вместе с ответом</span></footer></article>
          <aside className="evidence-rail" aria-label="Источники отчёта"><h2>Источники ответа</h2><p>Сохранённые фрагменты резюме для проверки выводов.</p><div className="evidence-sources">{citedSources(reply).map(({source, number}) => <button className="evidence-link" key={number} onClick={() => setSelection({source, number, messageId: reply.id})}><span className="evidence-index">[{number}]</span><span><strong>{source.name}</strong><small>Сохранённый источник</small></span></button>)}</div>
            {result?.questions.length > 0 && <section className="interview-note"><h3>Вопросы для интервью</h3><ul>{result.questions.map((q, i) => <li className="fact-copy" key={i}><MessageMarkdown message={reply} text={q} onSource={setSelection} /></li>)}</ul></section>}</aside>
        </div></>;
  }

  if (loading) return <main className="app-loading"><Icon name="mark" /><h1>CV Screener</h1><p>Загружаем базу и историю подборов…</p></main>;
  if (!boot) return <main className="empty"><h1>Сервис недоступен</h1><p role="alert">{error}</p><button className="button" onClick={() => location.reload()}>Повторить загрузку</button></main>;
  const title = viewIs('saved') ? 'Избранные кандидаты' : viewIs('library') ? 'База резюме' : reply?.analysis?.title || conversation?.title || (isReport ? 'Отчёт по подбору' : 'Новый подбор');
  return <NavigationContext.Provider value={{href,navigate}}>
    <a className="skip" href="#main">К содержимому</a>
    {error && <div className="error-box request-error" id="request-error" role="alert">{error}<button aria-label="Закрыть сообщение об ошибке" onClick={() => setError('')}><Icon name="close" /></button></div>}
    <div className="workspace-shell"><aside className="sidebar" aria-label="Навигация и история">
      <Nav changes={{id: null, turn: null, view: 'candidates'}} className="brand"><span className="brand-mark"><Icon name="mark" /></span><span translate="no">CV Screener</span></Nav>
      <button className="new-search" onClick={() => {navigate({id: null, turn: null, view: 'candidates'}); setTimeout(() => input.current?.focus(), 0);}}><Icon name="plus" /> Новый подбор</button>
      <nav className="side-nav" aria-label="Разделы"><Nav changes={{view: 'candidates'}} className={`side-link ${!['saved', 'library'].includes(route.view) ? 'active' : ''}`} aria-label="Подборы"><Icon name="folder" /><span className="nav-label">Подборы</span></Nav>
        <Nav changes={{view: 'saved'}} className={`side-link ${viewIs('saved') ? 'active' : ''}`} aria-label="Избранное"><Icon name="star" /><span className="nav-label">Избранное</span><span className="count">{favorites.length}</span></Nav>
        <Nav changes={{view: 'library'}} className={`side-link ${viewIs('library') ? 'active' : ''}`} aria-label="База резюме"><Icon name="file" /><span className="nav-label">База резюме</span><span className="count">{boot.library.length}</span></Nav></nav>
      <div className="history-group"><p className="side-section-title">История подборов</p><nav className="history" aria-label="Сохранённые диалоги">{threads.map(t => <Nav changes={{id: t.id, turn: null, view: isReport ? 'report' : 'candidates'}} className={t.id === route.id ? 'active' : ''} aria-current={t.id === route.id ? 'page' : undefined} key={t.id}><span className="history-title" title={t.title}>{t.title}</span><small>{date(t.updated_at)}</small></Nav>)}{!threads.length && <p className="history-empty">Здесь появятся ваши подборы</p>}</nav></div>
      <div className="sidebar-bottom"><span className={`status-dot ${boot.search_available ? '' : 'offline'}`} /><div><strong>{boot.search_available ? 'База подключена' : 'База недоступна'}</strong>{boot.indexed_count ?? '—'} резюме в поиске</div></div>
    </aside><main className="workspace-main" id="main" tabIndex={-1}>
      <div className="mobile-history"><label htmlFor="mobile-history">История подборов</label><select id="mobile-history" name="history" value={route.id || ''} onChange={event => navigate({id:event.target.value || null,turn:null,view:isReport && event.target.value ? 'report' : 'candidates'})}><option value="">Новый подбор</option>{threads.map(t => <option key={t.id} value={t.id}>{t.title}</option>)}</select></div>
      <div className="page-heading"><div><h1>{title}</h1><p>{viewIs('library') ? 'Исходные файлы из вашей базы.' : viewIs('saved') ? 'Сохранённые кандидаты и факты из резюме.' : isReport ? 'Выводы, сравнение и источники выбранного ответа.' : conversation ? 'Сведения из резюме. Ссылки открывают первоисточники.' : 'Поиск и сравнение по вашей базе резюме.'}</p></div><div className="heading-actions">{reply && (isReport ? <a className="button primary" href={`/api/conversations/${conversation.id}/report/${reply.id}`} download><Icon name="download" /> Скачать отчёт</a> : <Nav changes={{view: 'report'}} className="button"><Icon name="file" /> Открыть отчёт</Nav>)}</div></div>
      {conversation && !['saved', 'library'].includes(route.view) && <div className="results-top"><nav className="view-tabs" aria-label="Вид результатов">
        <Nav changes={{view: 'candidates'}} className={viewIs('candidates') ? 'active' : ''} aria-current={viewIs('candidates') ? 'page' : undefined}>Кандидаты <span className="number-pill">{reply?.analysis?.candidates.length || 0}</span></Nav>
        <Nav changes={{view: 'compare'}} className={viewIs('compare') ? 'active' : ''} aria-current={viewIs('compare') ? 'page' : undefined}>Сравнение</Nav>
        <Nav changes={{view: 'chat'}} className={viewIs('chat') ? 'active' : ''} aria-current={viewIs('chat') ? 'page' : undefined}>Диалог <span className="number-pill">{conversation.messages.length}</span></Nav>
        <Nav changes={{view: 'report'}} className={isReport ? 'active' : ''} aria-current={isReport ? 'page' : undefined}>Отчёт</Nav>
      </nav></div>}
      {!['report', 'saved', 'library'].includes(route.view) && renderQuery()}
      {isReport ? renderReport() : renderResults()}
      <p className="workspace-footnote"><Icon name="info" /> История хранится на этом компьютере. Для ответа найденные фрагменты резюме передаются OpenAI.</p>
    </main></div>
    <SourceDialog selection={selection} onClose={() => setSelection(null)} />
    <div className={`toast ${toast ? 'visible' : ''}`} role="status" aria-live="polite">{toast}</div>
  </NavigationContext.Provider>;
}
