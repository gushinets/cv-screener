import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
with patch('search.search.Search'):
    import analysis
    import cv_screener as core
    from server import create_app
from history_store import HistoryStore


class HistoryContractTests(unittest.TestCase):
    def test_database_contains_only_current_history_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.sqlite3'
            store = HistoryStore(path)
            with store.connect() as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(tables, {'conversations', 'requests', 'favorites'})

    def test_api_exposes_only_current_user_scenarios(self):
        with tempfile.TemporaryDirectory() as folder:
            app = create_app(HistoryStore(Path(folder) / 'history.sqlite3'))
            routes = {(method, route.path) for route in app.routes if route.path.startswith('/api/')
                      for method in route.methods}
            self.assertEqual(routes, {
                ('GET', '/api/bootstrap'), ('GET', '/api/conversations'),
                ('GET', '/api/conversations/{conversation_id}'), ('POST', '/api/chat'),
                ('GET', '/api/resume-imports'), ('POST', '/api/resume-imports'),
                ('GET', '/api/resume'), ('GET', '/api/favorites'), ('POST', '/api/favorites'),
                ('DELETE', '/api/favorites/{name}'),
                ('GET', '/api/conversations/{conversation_id}/report/{message_id}'),
            })

    def test_model_history_keeps_native_snapshots_and_remaps_links_without_mutating_them(self):
        source = {'name': 'Emily.txt', 'text': 'Emily built EKS.'}
        turn = {'role': 'assistant', 'content': 'Emily [1]', 'sources': [source], 'analysis': None}
        history = analysis.history_for_model([turn])
        self.assertEqual(set(history[0]), {'role', 'content', 'sources'})
        self.assertEqual(history[0]['sources'], [source])
        current = [{'name': 'Bob.txt', 'text': 'Bob uses Python.'}, source]
        messages = core.model_messages('Compare them', history, current)
        self.assertEqual(messages[1], {'role': 'assistant', 'content': 'Emily [2]'})
        self.assertEqual(turn['content'], 'Emily [1]')
        self.assertEqual(history[0]['sources'], [source])


if __name__ == '__main__':
    unittest.main()
