import React, {useEffect, useRef, useState} from 'react';
import {api} from './api.js';
import Icon from './Icons.jsx';

const labels = {queued: 'Ожидает индексации', extracting: 'Читаем резюме', indexing: 'Добавляем в поиск', ready: 'В поиске', error: 'Ошибка обработки'};
const pending = job => !['ready', 'error'].includes(job.status);

export default function ResumeLibrary({library, onRefresh, onSource}) {
  const [file, setFile] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [sending, setSending] = useState(false);
  const [checking, setChecking] = useState(true);
  const [version, setVersion] = useState(0);
  const [error, setError] = useState('');
  const [pollError, setPollError] = useState('');
  const input = useRef(null);
  const completed = useRef(null);
  const busy = sending || jobs.some(pending);

  useEffect(() => {
    let ignore = false, timer;
    async function check() {
      let again = true;
      try {
        const values = await api('/resume-imports');
        if (ignore) return;
        setJobs(values); setPollError(''); setChecking(false);
        const signature = values.filter(job => !pending(job)).map(job => `${job.id}:${job.status}`).join();
        if (signature !== completed.current) {
          const settings = await api('/bootstrap');
          if (ignore) return;
          onRefresh(settings); completed.current = signature;
        }
        again = values.some(pending);
      } catch (err) { if (!ignore) setPollError(err.message); }
      if (!ignore && again) timer = setTimeout(check, 1000);
    }
    check();
    return () => { ignore = true; clearTimeout(timer); };
  }, [version, onRefresh]);

  async function upload(event) {
    event.preventDefault();
    if (!file || busy || checking) return;
    if (!/\.(pdf|docx|txt)$/i.test(file.name)) { setError('Выберите PDF, DOCX или TXT.'); return; }
    if (file.size > 10 * 1024 * 1024) { setError('Максимальный размер резюме — 10 МБ.'); return; }
    setSending(true); setError('');
    try {
      const body = new FormData(); body.append('file', file);
      const job = await api('/resume-imports', {method: 'POST', body});
      setJobs(old => [job, ...old.filter(item => item.id !== job.id)]);
      setFile(null); input.current.value = '';
    } catch (err) { setError(err.message); }
    finally { setSending(false); setVersion(value => value + 1); }
  }

  async function open(name) {
    try { const source = await api(`/resume?name=${encodeURIComponent(name)}`); onSource({source, number: null}); }
    catch (err) { setError(err.message); }
  }

  return <>
    <form className="resume-upload" onSubmit={upload} aria-label="Загрузка резюме" aria-busy={busy}>
      <div className="upload-heading"><span className="upload-symbol"><Icon name="upload" /></span><div><h2>Добавить резюме</h2><p>После обработки оно сразу участвует в подборе.</p></div></div>
      <div className="upload-controls"><label className="file-picker"><Icon name="file" /><span>{file?.name || 'Выбрать файл'}</span>
        <input ref={input} className="sr-only" type="file" name="resume" aria-label="Файл резюме" aria-describedby="upload-help" accept=".pdf,.docx,.txt" disabled={busy || checking} onChange={event => {setFile(event.target.files[0] || null); setError('');}} />
      </label><button className="button primary" disabled={!file || busy || checking || !!pollError}><Icon name="upload" />{sending ? 'Загружаем…' : 'Загрузить и добавить'}</button></div>
      <p id="upload-help" className="upload-help">PDF с текстовым слоем, DOCX или TXT. До 10 МБ. Существующие файлы не перезаписываются.</p>
      {error && <p className="upload-error" role="alert">{error}</p>}
      {pollError && <p className="upload-error" role="alert">{pollError} Статус обновится после восстановления соединения.</p>}
    </form>
    {jobs.length > 0 && <section className="import-status" aria-label="Обработка резюме"><h2>Последние загрузки</h2><ul aria-live="polite">{jobs.slice(0, 5).map(job => <li key={job.id}>
      <div className="import-file"><Icon name={job.status === 'ready' ? 'check' : 'file'} /><strong>{job.name}</strong></div>
      <span className={`import-label ${job.status}`}>{labels[job.status]}</span>
      {job.error && <p className="import-error">{job.error}</p>}
      {job.status === 'ready' && <button className="text-button" type="button" onClick={() => open(job.name)}>Открыть резюме</button>}
    </li>)}</ul></section>}
    <section className="library-list" aria-label="Документы базы">{library.map(item => <div className="library-item" key={item.name}><Icon name="file" /><span>{item.name}<em className={`library-index-state ${item.indexed ? 'indexed' : ''}`}>{item.indexed ? 'В поиске' : 'Не в поиске'}</em></span><small>{item.format || 'TXT'}</small><button className="text-button" onClick={() => open(item.name)}>Открыть <Icon name="arrow" /></button></div>)}</section>
  </>;
}
