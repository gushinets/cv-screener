"""Loopback API and React UI, sharing the existing Qdrant/OpenAI pipeline."""
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

import analysis
import cv_screener as core
from history_store import HistoryStore
from resumes import MAX_BYTES, ResumeError, extract_text, index_resume, library_files, read_resume, resume_id, validate_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


class UploadBodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/api/resume-imports' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > MAX_BYTES + 64 * 1024:
                raise HTTPException(413, 'Максимальный размер резюме — 10 МБ.')
            return message

        await self.app(scope, limited_receive, send)


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    revision: int = Field(default=0, ge=0)
    request_id: UUID
    message: str = Field(min_length=1, max_length=12000)


class FavoriteRequest(BaseModel):
    conversation_id: UUID
    message_id: UUID
    source_id: int = Field(ge=1)


def resume_path(name):
    try:
        validate_name(name)
    except ResumeError:
        raise HTTPException(404, 'Резюме не найдено.') from None
    path = (DATA / name).resolve()
    if path.parent != DATA.resolve() or not path.is_file() or (DATA / name).is_symlink():
        raise HTTPException(404, 'Исходный файл резюме недоступен. Сохранённые фрагменты остаются в истории.')
    return path


def export_report(conversation, message):
    result = message.get('analysis')
    sources = message.get('sources', [])

    def cited(text, source_id=None):
        if source_id is not None and f'[{source_id}]' not in text:
            text += f' [{source_id}]'
        return re.sub(r'\[(\d+)\](?!\()',
                      lambda m: f'[{m[1]}](#source-{m[1]})' if 1 <= int(m[1]) <= len(sources) else m[0],
                      escape(text, quote=False))

    title = (result or {}).get('title') or conversation['title']
    lines = [f'# {escape(title, quote=False)}', '', cited(message['content']), '']
    if result:
        lines.extend(['## Обоснование', cited(result['recommendation']), ''])
        if result['candidates']:
            lines.append('## Кандидаты')
        for candidate in result['candidates']:
            lines.extend(['', '### ' + cited(candidate['name'], candidate['source_id']), cited(candidate['summary'])])
            for fact in candidate['facts']:
                lines.extend(['- ' + cited(fact['text'], candidate['source_id']),
                              '> ' + escape(fact['quote'], quote=False).replace('\n', '\n> ')])
            lines.extend('- Уточнить: ' + cited(gap) for gap in candidate['gaps'])
        if result['criteria']:
            lines.extend(['', '## Сравнение'])
        for criterion in result['criteria']:
            lines.extend(['', '### ' + cited(criterion['title'])])
            for cell in criterion['cells']:
                name = sources[cell['source_id'] - 1]['name']
                lines.append('- ' + cited(f"{name}: {cell['text']}", cell['source_id']))
        if result['questions']:
            lines.extend(['', '## Вопросы для интервью', *['- ' + cited(q) for q in result['questions']]])
    lines.extend(['', '## Сохранённые источники'])
    for index, source in enumerate(sources, 1):
        fence = '`' * max(3, 1 + max((len(m[0]) for m in re.finditer(r'`+', source['text'])), default=0))
        lines.extend(['', f'<a id="source-{index}"></a>', f"### [{index}] {escape(source['name'], quote=False)}",
                      fence + 'text', source['text'], fence])
    return '\n'.join(lines)


def create_app(store=None):
    app = FastAPI(title='CV Screener', docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])
    app.add_middleware(UploadBodyLimit)
    history = store or HistoryStore()
    # shortcut: one local worker serializes generation; use per-conversation jobs for shared deployments.
    generation_lock = asyncio.Lock()
    # shortcut: recent processing statuses belong to this worker; persist jobs for resumable multi-worker imports.
    imports = {}

    async def process_resume(job, content, path):
        created = False
        store = None
        # Searches must also wait for rollback if a new index fails partway through.
        await generation_lock.acquire()
        try:
            job['status'] = 'extracting'
            text = await asyncio.to_thread(extract_text, content, path.suffix)
            job.update(status='indexing', characters=len(text))
            store = core.search.getStore()
            with path.open('xb') as output:
                created = True
                output.write(content)
            await index_resume(store, path, text)
            job['status'] = 'ready'
        except Exception as error:
            message = str(error) if isinstance(error, ResumeError) else 'Не удалось добавить резюме. Проверьте Qdrant и повторите загрузку.'
            if isinstance(error, FileExistsError):
                message = 'Файл с таким именем уже существует. Переименуйте новое резюме.'
            if created:
                try:
                    await store.aremove([resume_id(path)])
                except Exception:
                    logging.exception('Cannot roll back resume index')
                    message = 'Не удалось подтвердить индексацию. Оригинал сохранён; восстановите Qdrant и повторите импорт из командной строки.'
                else:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logging.exception('Cannot remove failed resume file')
                        message = 'Индексация не завершена. Не удалось удалить файл после ошибки; проверьте права на папку data.'
            if not isinstance(error, (ResumeError, FileExistsError)):
                logging.exception('Resume import failed')
            job.update(status='error', error=message)
        finally:
            generation_lock.release()

    @app.middleware('http')
    async def local_requests(request: Request, call_next):
        if request.url.path.startswith('/api/'):
            origin = request.headers.get('origin')
            if origin and origin != f'{request.url.scheme}://{request.headers.get("host")}':
                return JSONResponse({'detail': 'Разрешены только запросы из локального интерфейса.'}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/bootstrap')
    async def bootstrap():
        paths = {}
        try:
            paths = await core.search.getStore().agetPaths(limit=1000)
            indexed, available = len(paths), True
        except Exception:
            indexed, available = None, False
        return {'model': core.chat_model, 'indexed_count': indexed, 'search_available': available,
                'examples': core.EXAMPLES,
                'library': [{'name': p.name, 'size': p.stat().st_size, 'format': p.suffix[1:].upper(),
                             'indexed': p.name in paths} for p in library_files(DATA)]}

    @app.get('/api/resume-imports')
    def resume_imports():
        return list(reversed(list(imports.values())))

    @app.post('/api/resume-imports', status_code=202)
    async def upload_resume(file: UploadFile, background_tasks: BackgroundTasks):
        try:
            name = validate_name(file.filename)
        except ResumeError as error:
            raise HTTPException(422, str(error)) from None
        if any(job['status'] not in ('ready', 'error') for job in imports.values()):
            raise HTTPException(409, 'Дождитесь завершения текущей обработки и загрузите следующее резюме.')
        content = await file.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, 'Максимальный размер резюме — 10 МБ.')
        path = DATA / name
        if path.resolve().parent != DATA.resolve() or path.exists() or path.is_symlink():
            raise HTTPException(409, 'Файл с таким именем уже есть в базе. Переименуйте новое резюме.')
        if any(job['status'] not in ('ready', 'error') for job in imports.values()):
            raise HTTPException(409, 'Дождитесь завершения текущей обработки и загрузите следующее резюме.')
        job = {'id': str(uuid4()), 'name': name, 'status': 'queued', 'created_at': datetime.now(timezone.utc).isoformat()}
        for key in list(imports)[:-19]:
            del imports[key]
        imports[job['id']] = job
        background_tasks.add_task(process_resume, job, content, path)
        return dict(job)

    @app.get('/api/conversations')
    def conversations():
        return history.list()

    @app.get('/api/conversations/{conversation_id}')
    def get_conversation(conversation_id: UUID):
        result = history.get(str(conversation_id))
        if result is None:
            raise HTTPException(404, 'Диалог не найден.')
        return result


    @app.post('/api/chat')
    async def chat(body: ChatRequest):
        query = body.message.strip()
        if not query:
            raise HTTPException(422, 'Опишите вакансию или задайте вопрос о кандидатах.')
        async with generation_lock:
            replay = history.replay(str(body.request_id))
            if replay:
                return replay
            current = history.get(str(body.conversation_id)) if body.conversation_id else None
            if body.conversation_id and current is None:
                raise HTTPException(404, 'Диалог не найден.')
            if (current['revision'] if current else 0) != body.revision:
                raise HTTPException(409, 'Диалог изменился. Обновите его перед отправкой.')
            try:
                answer = await analysis.analyze(query, current['messages'] if current else [])
                return history.append(str(body.conversation_id) if body.conversation_id else None,
                                      body.revision, query, answer, str(body.request_id))
            except AuthenticationError:
                raise HTTPException(502, 'OpenAI отклонил API-ключ. Обновите ключ в настройках сервера.') from None
            except RateLimitError:
                raise HTTPException(429, 'OpenAI: превышен лимит запросов или исчерпан баланс API.') from None
            except APIConnectionError:
                raise HTTPException(502, 'Не удалось подключиться к OpenAI. Проверьте интернет и повторите запрос.') from None
            except APIStatusError:
                raise HTTPException(502, 'OpenAI не обработал запрос. Проверьте доступность настроенной модели.') from None
            except ValueError as error:
                raise HTTPException(422, str(error)) from None

    @app.get('/api/resume')
    def resume(name: str):
        try:
            return {'name': name, 'text': read_resume(resume_path(name))}
        except ResumeError as error:
            raise HTTPException(422, str(error)) from None

    @app.get('/api/favorites')
    def favorites():
        return history.favorites()

    @app.post('/api/favorites')
    def add_favorite(body: FavoriteRequest):
        conversation = history.get(str(body.conversation_id))
        message = next((m for m in (conversation or {}).get('messages', []) if m['id'] == str(body.message_id)), None)
        candidate = next((c for c in (message or {}).get('analysis', {}).get('candidates', [])
                          if c['source_id'] == body.source_id), None) if message and message.get('analysis') else None
        if candidate is None or body.source_id > len(message.get('sources', [])):
            raise HTTPException(404, 'Карточка кандидата не найдена в этом ответе.')
        source = message['sources'][body.source_id - 1]
        history.favorite(source['name'], {'name': source['name'], 'source': source, 'candidate': candidate,
                                         'conversation_id': str(body.conversation_id), 'message_id': str(body.message_id)})
        return history.favorites()

    @app.delete('/api/favorites/{name}')
    def remove_favorite(name: str):
        history.favorite(name, None)
        return history.favorites()

    @app.get('/api/conversations/{conversation_id}/report/{message_id}')
    def report(conversation_id: UUID, message_id: UUID):
        conversation = history.get(str(conversation_id))
        message = next((m for m in (conversation or {}).get('messages', [])
                        if m['id'] == str(message_id) and m['role'] == 'assistant'), None)
        if message is None:
            raise HTTPException(404, 'Ответ для отчёта не найден.')
        return Response(export_report(conversation, message), media_type='text/markdown; charset=utf-8',
                        headers={'Content-Disposition': 'attachment; filename="cv-screener-report.md"'})

    dist = ROOT / 'frontend' / 'dist'
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
    return app


if __name__ == '__main__':
    uvicorn.run(create_app(), host='127.0.0.1', port=7860, log_level='info')
