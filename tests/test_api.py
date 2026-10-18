import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
with patch('search.search.Search'):
    import analysis
    import cv_screener as core
    from server import create_app, export_report
from history_store import HistoryStore


SOURCE = {'name': 'Emily.txt', 'text': 'Emily Roberts\nBuilt EKS with Terraform.\nProject [99] is documented.'}


def result():
    return analysis.Analysis(
        title='DevOps', answer='Emily Roberts подходит [1].', recommendation='Обсудить опыт EKS [1].',
        recommended_source_id=1,
        candidates=[analysis.Candidate(source_id=1, name='Emily Roberts', role='DevOps', summary='EKS [1]',
                    facts=[analysis.Fact(text='EKS и Terraform', quote='Built EKS with Terraform.')], gaps=['Размер команды'])],
        criteria=[analysis.Criterion(title='Terraform', cells=[analysis.Cell(source_id=1, text='Есть опыт', status='confirmed')])],
        questions=['Как проверяете изменения?'])


def answer():
    r = result()
    return {'content': r.answer, 'sources': [dict(SOURCE)], 'analysis': r.model_dump(), 'model': 'test-model'}


class HistoryTests(unittest.TestCase):


    def test_append_conflict_rolls_back_and_report_uses_original_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HistoryStore(Path(folder) / 'history.sqlite3')
            first = store.append(None, 0, 'DevOps', answer(), str(uuid4()))
            with self.assertRaises(ValueError):
                store.append(first['id'], 0, 'Stale query', answer(), str(uuid4()))
            restored = store.get(first['id'])
            self.assertEqual(len(restored['messages']), 2)
            self.assertEqual(restored['revision'], 1)
            report = export_report(restored, restored['messages'][1])
            self.assertIn('[1] Emily.txt', report)
            self.assertIn(SOURCE['text'], report)


    def test_report_uses_selected_title_and_links_without_duplicate_references(self):
        message = answer()
        message['analysis']['title'] = 'Selected comparison'
        message['analysis']['candidates'][0]['facts'][0]['text'] = 'EKS and Terraform [1].'
        report = export_report({'title': 'Original selection'}, message)
        self.assertTrue(report.startswith('# Selected comparison\n'))
        self.assertIn('- EKS and Terraform [1](#source-1).\n', report)
        self.assertIn('<a id="source-1"></a>', report)
        self.assertIn(SOURCE['text'], report)
        self.assertNotIn('[1]. [1]', report)

    def test_report_preserves_literal_sources_and_existing_links(self):
        message = answer()
        source = 'Resume with ``` inside\n<script>literal source text</script>'
        message['sources'][0]['text'] = source
        message['content'] = 'Source [1], unknown [99], external [1](https://example.com).'
        message['analysis']['candidates'][0]['facts'][0]['quote'] = '<literal>\nnext line'
        report = export_report({'title': 'Original'}, message)
        self.assertIn('````text\n' + source + '\n````', report)
        self.assertIn('unknown [99]', report)
        self.assertIn('external [1](https://example.com)', report)
        self.assertIn('> &lt;literal&gt;\n> next line', report)

    def test_report_without_candidates_keeps_explanation_and_sources_without_empty_sections(self):
        message = answer()
        message['content'] = 'No matching candidates [1].'
        message['analysis'].update(candidates=[], criteria=[], questions=[], recommended_source_id=None,
                                   recommendation='No matching candidates [1].')
        report = export_report({'title': 'No matches'}, message)
        for heading in ['## Кандидаты', '## Сравнение', '## Вопросы для интервью']:
            self.assertNotIn(heading, report)
        self.assertIn('No matching candidates [1](#source-1).', report)
        self.assertIn(SOURCE['text'], report)


class StructuredTests(unittest.IsolatedAsyncioTestCase):
    def test_citation_ranges_are_rejected_without_changing_literal_quotes(self):
        source = {**SOURCE, 'text': SOURCE['text'] + '\nProject references [1]–[3].'}
        good = result(); good.candidates[0].facts[0].quote = 'Project references [1]–[3].'
        analysis.validate_analysis(good, [source] * 3)
        for field in ['answer', 'recommendation', 'fact', 'gap', 'cell', 'question']:
            bad = result()
            text = 'See sources [1]–[3].'
            if field in ('answer', 'recommendation'): setattr(bad, field, text)
            if field == 'fact': bad.candidates[0].facts[0].text = text
            if field == 'gap': bad.candidates[0].gaps = [text]
            if field == 'cell': bad.criteria[0].cells[0].text = text
            if field == 'question': bad.questions = [text]
            with self.subTest(field=field), self.assertRaises(ValueError):
                analysis.validate_analysis(bad, [source] * 3)

    def test_validation_checks_quote_identity_source_range_and_cell_completeness(self):
        good = result()
        analysis.validate_analysis(good, [SOURCE])
        good.candidates[0].facts[0].quote = 'Project [99] is documented.'
        analysis.validate_analysis(good, [SOURCE])
        for change in ('quote', 'range', 'cells', 'name'):
            bad = result()
            if change == 'quote': bad.candidates[0].facts[0].quote = 'Invented experience'
            if change == 'range': bad.candidates[0].source_id = 2
            if change == 'cells': bad.criteria[0].cells = []
            if change == 'name': bad.candidates[0].name = 'Someone Else'
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.validate_analysis(bad, [SOURCE])

    async def test_structured_reply_repairs_bad_quote_once_and_history_keeps_card_references(self):
        bad = result(); bad.candidates[0].facts[0].quote = 'Invented'
        responses = [SimpleNamespace(model='test-model', choices=[SimpleNamespace(message=SimpleNamespace(parsed=r, content=r.model_dump_json()))]) for r in [bad, result()]]
        with patch.object(core, 'retrieve_sources', AsyncMock(return_value=[SOURCE])), \
                patch.object(core.client.chat.completions, 'parse', AsyncMock(side_effect=responses)) as generate:
            reply = await analysis.analyze('DevOps', [])
        self.assertEqual(generate.await_count, 2)
        repair = generate.call_args.kwargs['messages'][-1]['content']
        self.assertIn('Emily Roberts ([1]) факт 1', repair)
        self.assertEqual(reply['analysis']['candidates'][0]['facts'][0]['quote'], 'Built EKS with Terraform.')
        turn = {'role': 'assistant', **reply}
        history = analysis.history_for_model([turn])
        self.assertIn('Emily Roberts [1]', history[0]['content'])
        self.assertEqual(history[0]['sources'], [SOURCE])


class ApiTests(unittest.TestCase):
    def test_request_id_prevents_duplicate_generation_and_bad_revision_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HistoryStore(Path(folder) / 'history.sqlite3')
            with TestClient(create_app(store), base_url='http://127.0.0.1') as client, \
                    patch('server.analysis.analyze', AsyncMock(return_value=answer())) as generate:
                body = {'message': 'DevOps', 'request_id': str(uuid4())}
                first = client.post('/api/chat', json=body)
                self.assertEqual(first.status_code, 200)
                self.assertEqual(client.post('/api/chat', json=body).json(), first.json())
                stale = client.post('/api/chat', json={**body, 'request_id': str(uuid4()), 'conversation_id': first.json()['id'], 'revision': 0})
                self.assertEqual(stale.status_code, 409)
                self.assertEqual(generate.await_count, 1)
                cid = first.json()['id']; mid = first.json()['messages'][1]['id']
                exported = client.get(f'/api/conversations/{cid}/report/{mid}')
                self.assertEqual(exported.status_code, 200)
                self.assertIn('attachment', exported.headers['content-disposition'])
                self.assertIn('Emily [1]', exported.text.replace('Emily Roberts', 'Emily'))
                self.assertEqual(client.post('/api/chat', json=body, headers={'Origin': 'https://unrelated.example'}).status_code, 403)
                self.assertEqual(client.get('/api/resume', params={'name': '../aiconfig.json'}).status_code, 404)
                self.assertEqual(client.get('/api/conversations').headers['cache-control'], 'no-store')
                fav = client.post('/api/favorites', json={'conversation_id': cid, 'message_id': mid, 'source_id': 1})
                self.assertEqual(fav.status_code, 200)
                self.assertEqual(fav.json()[0]['source'], SOURCE)

    def test_failed_generation_does_not_save_half_a_conversation(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HistoryStore(Path(folder) / 'history.sqlite3')
            with TestClient(create_app(store), base_url='http://127.0.0.1') as client, \
                    patch('server.analysis.analyze', AsyncMock(side_effect=ValueError('Invalid quote'))):
                response = client.post('/api/chat', json={'message': 'DevOps', 'request_id': str(uuid4())})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(store.list(), [])


if __name__ == '__main__':
    unittest.main()
