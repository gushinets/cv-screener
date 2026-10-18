"""Local, transactional conversation history and saved source snapshots."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DB_FILE = Path(__file__).resolve().parents[1] / '.local' / 'conversations.sqlite3'


def now():
    return datetime.now(timezone.utc).isoformat()


class HistoryStore:
    def __init__(self, path=DB_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, revision INTEGER NOT NULL, messages TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS favorites (name TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def list(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                'SELECT id,title,created_at,updated_at,revision FROM conversations ORDER BY updated_at DESC,id')]

    def get(self, conversation_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result['messages'] = json.loads(result['messages'])
            return result

    def replay(self, request_id):
        with self.connect() as db:
            row = db.execute('SELECT conversation_id FROM requests WHERE id=?', (request_id,)).fetchone()
        return self.get(row['conversation_id']) if row else None

    def append(self, conversation_id, revision, query, answer, request_id):
        conversation_id = conversation_id or str(uuid4())
        timestamp = now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if (old['revision'] if old else 0) != revision:
                raise ValueError('Диалог изменился в другой вкладке. Обновите его и повторите запрос.')
            messages = json.loads(old['messages']) if old else []
            messages.extend([{'id': str(uuid4()), 'role': 'user', 'content': query, 'created_at': timestamp},
                             {'id': str(uuid4()), 'role': 'assistant', 'created_at': timestamp, **answer}])
            title = old['title'] if old else ((answer.get('analysis') or {}).get('title') or query)[:100]
            db.execute('INSERT INTO conversations VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                       'updated_at=excluded.updated_at,revision=excluded.revision,messages=excluded.messages',
                       (conversation_id, title, old['created_at'] if old else timestamp, timestamp,
                        revision + 1, json.dumps(messages, ensure_ascii=False)))
            db.execute('INSERT INTO requests VALUES (?,?)', (request_id, conversation_id))
        return self.get(conversation_id)


    def favorites(self):
        with self.connect() as db:
            return [json.loads(row['snapshot']) for row in db.execute('SELECT snapshot FROM favorites ORDER BY name')]

    def favorite(self, name, snapshot):
        with self.connect() as db:
            if snapshot is None:
                db.execute('DELETE FROM favorites WHERE name=?', (name,))
            else:
                db.execute('INSERT INTO favorites VALUES (?,?) ON CONFLICT(name) DO UPDATE SET snapshot=excluded.snapshot',
                           (name, json.dumps(snapshot, ensure_ascii=False)))
