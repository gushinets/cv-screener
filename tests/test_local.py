import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from common.schema import Doc, DocFilter
from providers.store.qdrant import Store
from qdrant_client import models
import import_resumes

with patch('search.search.Search'):
    import cv_screener


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_resume_filenames_survive_semantic_rank_out(self):
        named = [Doc(page_content='Project manager', metadata={'parent': 'John Doe.txt'}),
                 Doc(page_content='QA engineer', metadata={'parent': 'Sr QA Resume.txt'})]
        noise = [Doc(page_content='QA', metadata={'parent': f'noise-{i}.txt'}) for i in range(30)]
        store = SimpleNamespace(agetPaths=AsyncMock(return_value={
            'John Doe.txt': 'pm', 'Sr QA Resume.txt': 'qa', 'Other.txt': 'other'}),
            aget=AsyncMock(return_value=named))
        with patch.object(cv_screener.search, 'asearch', AsyncMock(return_value=noise)), \
                patch.object(cv_screener.search, 'getStore', return_value=store):
            docs = await cv_screener.query_qdrant('Сравни JOHN DOE.TXT и Sr QA Resume.txt')
        self.assertEqual([doc.metadata['parent'] for doc in docs[:2]], ['John Doe.txt', 'Sr QA Resume.txt'])
        self.assertEqual(len(docs), 25)
        self.assertEqual(store.aget.call_args.args[0].objectIds, ['pm', 'qa'])

    async def test_query_reranks_a_wider_pool_and_limits_distinct_resumes(self):
        docs = [Doc(page_content='Python', metadata={'parent': 'same.txt', 'chunk': index}) for index in range(5)]
        docs.extend(Doc(page_content='Python', metadata={'parent': f'cv-{index}.txt'}) for index in range(30))
        with patch.object(cv_screener.search, 'asearch', AsyncMock(return_value=docs)) as search:
            selected = await cv_screener.query_qdrant('Python')
        self.assertEqual(len(selected), 25)
        self.assertEqual(len({doc.metadata['parent'] for doc in selected}), 25)
        self.assertIs(selected[0], docs[0])
        self.assertEqual(search.call_args.kwargs['docFilter'].limit, 100)
        self.assertTrue(search.call_args.kwargs['docFilter'].useQuickRank)

    async def test_query_points_returns_document_and_uses_embedding(self):
        point = models.ScoredPoint(
            id=1, version=0, score=0.9,
            payload={'content': 'Python developer', 'meta': {'parent': 'resume.txt', 'chunk': 0}})
        store = Store.__new__(Store)
        store._collection = 'test'
        store.encodeString = Mock(return_value=[0.1, 0.2])
        store._client = SimpleNamespace(
            query_points=AsyncMock(return_value=SimpleNamespace(points=[point])))
        docs = await store.asearch('Python', docFilter=DocFilter())
        self.assertEqual(docs[0].page_content, 'Python developer')
        self.assertEqual(docs[0].score, 0.9)
        self.assertEqual(store._client.query_points.call_args.kwargs['query'], [0.1, 0.2])


    def test_generation_parameters_for_configured_model(self):
        for model, config, expected in [
                ('gpt-6-luna', {'reasoning_effort': 'medium'}, {'reasoning_effort': 'medium'}),
                ('gpt-6-luna', {'reasoning_effort': 'low'}, {'reasoning_effort': 'low'}),
                ('gpt-6-luna', {}, {'reasoning_effort': 'medium'}),
                ('gpt-3.5-turbo-16k', {'temperature': 0.2}, {'temperature': 0.2})]:
            with self.subTest(model=model), patch.object(cv_screener, 'chat_model', model), \
                    patch.object(cv_screener, 'chat_config', config):
                self.assertEqual(cv_screener.generation_parameters(), expected)

    async def test_followup_keeps_only_cited_sources_and_merges_previous_fragments(self):
        previous = [{'name': 'Emily.txt', 'text': 'Emily: DevOps engineer'},
                    {'name': 'Unused.txt', 'text': 'Unrelated resume'},
                    {'name': 'Bob.txt', 'text': 'Old Bob fragment'}]
        history = [{'role': 'assistant', 'content': 'Emily [1]; Bob [3]', 'sources': previous}]
        doc = Doc(page_content='Current Bob fragment', metadata={'parent': 'Bob.txt'})
        with patch.object(cv_screener, 'query_qdrant', AsyncMock(return_value=[doc])):
            sources = await cv_screener.retrieve_sources('Compare them', history)
        self.assertEqual(sources, [
            {'name': 'Bob.txt', 'text': 'Current Bob fragment\n\nOld Bob fragment'}, previous[0]])
        messages = cv_screener.model_messages('Compare them', history, sources)
        self.assertIn('[2] Emily.txt\nEmily: DevOps engineer', messages[0]['content'])
        self.assertNotIn('Unused.txt', messages[0]['content'])
        self.assertEqual(messages[1]['content'], 'Emily [2]; Bob [1]')
        self.assertEqual(history[0]['content'], 'Emily [1]; Bob [3]')

    async def test_followup_retains_earlier_shortlist_and_all_cited_evidence(self):
        earlier = [{'name': 'Emily.txt', 'text': 'Emily deployed EKS clusters.\nEmily holds CKA.'},
                   {'name': 'Michael.txt', 'text': 'Michael led 15 engineers.'},
                   {'name': 'Unused.txt', 'text': 'Not cited'}]
        history = [
            {'role': 'assistant', 'content': 'Emily [1]; reserve Michael [2]',
             'sources': earlier},
            {'role': 'assistant', 'content': 'Emily holds CKA [1]',
             'sources': [{'name': 'Emily.txt', 'text': 'Emily holds CKA.'}]},
        ]
        unchanged = json.dumps(history)
        docs = [Doc(page_content='Emily holds CKA.', metadata={'parent': 'Emily.txt'})]
        with patch.object(cv_screener, 'query_qdrant', AsyncMock(return_value=docs)):
            sources = await cv_screener.retrieve_sources('Compare the original two candidates', history)
        self.assertIn('Emily deployed EKS clusters.', sources[0]['text'])
        self.assertEqual(sources[0]['text'].count('Emily holds CKA.'), 1)
        self.assertEqual(sources[1], earlier[1])
        self.assertNotIn('Unused.txt', [source['name'] for source in sources])
        self.assertEqual(json.dumps(history), unchanged)

    async def test_retrieval_loads_all_indexed_resume_chunks_in_document_order(self):
        def chunk(name, object_id, number, text):
            return Doc(page_content=text, metadata={'parent': name, 'objectId': object_id, 'chunk': number})
        matches = [chunk('ML.txt', 'ml', 0, 'ML skills: Python'),
                   chunk('Alex.txt', 'alex', 1, 'Python automation')]
        stored = [chunk('Alex.txt', 'alex', 1, 'Python automation'),
                  chunk('ML.txt', 'ml', 1, 'Built production REST APIs with PyTorch'),
                  chunk('Alex.txt', 'alex', 0, 'Alex profile'), matches[0]]
        store = SimpleNamespace(aget=AsyncMock(return_value=stored))
        with patch.object(cv_screener, 'query_qdrant', AsyncMock(return_value=matches)), \
                patch.object(cv_screener.search, 'getStore', return_value=store):
            sources = await cv_screener.retrieve_sources('Python backend', [])
        self.assertEqual(sources, [
            {'name': 'ML.txt', 'text': 'ML skills: Python\n\nBuilt production REST APIs with PyTorch'},
            {'name': 'Alex.txt', 'text': 'Alex profile\n\nPython automation'}])
        query = store.aget.call_args.args[0]
        self.assertEqual(query.objectIds, ['ml', 'alex'])
        self.assertFalse(query.isTable)
        self.assertEqual(matches[0].page_content, 'ML skills: Python')

    def test_citations_require_existing_numeric_source_ids(self):
        for answer, count, expected in [
            ('Candidate [1]', 1, True), ('Candidates [1] and [2]', 2, True),
            ('Candidate [N]', 2, False), ('Candidate [1] [N]', 2, False),
            ('Candidates [1]–[2]', 2, False), ('Candidates [1] and [1-2]', 2, False),
            ('Candidate [0]', 2, False), ('Candidate [3]', 2, False),
            ('Candidate without a citation', 2, False),
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(bool(cv_screener.valid_citations(answer, count)), expected)

    def test_history_citations_follow_source_identity_when_retrieval_order_changes(self):
        previous = [{'name': 'Emily.txt', 'text': 'DevOps experience'}]
        current = [{'name': 'Bob.txt', 'text': 'Other experience'}, *previous]
        turn = {'role': 'assistant', 'content': 'Emily [1]',
                'sources': previous}
        self.assertEqual(cv_screener.history_content(turn, current),
                         'Emily [2]')
        self.assertEqual(turn['content'], 'Emily [1]')
        self.assertEqual(cv_screener.history_content(turn, current[:1]),
                         'Emily (источник: Emily.txt)')
        uncited = {'role': 'assistant', 'content': 'Old response [1] [N]'}
        self.assertEqual(cv_screener.history_content(uncited, current), 'Old response  ')
        user = {'role': 'user', 'content': 'Explain source [1]'}
        self.assertEqual(cv_screener.history_content(user, current), user['content'])


    async def test_import_uses_stable_ids_and_closes_client(self):
        store = SimpleNamespace(aadd=AsyncMock(), acount_documents=AsyncMock(return_value=1),
                                _client=SimpleNamespace(close=AsyncMock()))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'resume.txt').write_text('Python developer', encoding='utf-8')
            with patch.object(import_resumes, 'getStore', return_value=store):
                await import_resumes.import_resumes(folder)
                await import_resumes.import_resumes(folder)
        first, second = store.aadd.call_args_list
        self.assertEqual(first.kwargs['objectId'], second.kwargs['objectId'])
        self.assertEqual(first.kwargs['parent'], 'resume.txt')
        self.assertEqual(store._client.close.await_count, 2)

    async def test_empty_resume_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'empty.txt').write_text(' ', encoding='utf-8')
            with patch.object(import_resumes, 'getStore') as get_store:
                with self.assertRaisesRegex(ValueError, 'empty.txt'):
                    await import_resumes.import_resumes(folder)
                get_store.assert_not_called()

    def test_existing_collection_with_wrong_dimension_is_preserved(self):
        config = {'store': 'qdrant', 'qdrant': {
            'collection': 'test', 'similarity': 'Cosine', 'renderChunkSize': 1000,
            'host': 'localhost', 'port': 6333}}
        client = Mock()
        client.get_collection.return_value.config.params.vectors = models.VectorParams(
            size=99, distance=models.Distance.COSINE)
        with patch('providers.store.qdrant.DocumentStoreBase.__init__', return_value=None), \
                patch('providers.store.qdrant.Config.getConfig', return_value=config), \
                patch.object(Store, 'getVectorSize', return_value=384), \
                patch('providers.store.qdrant.QdrantClient', return_value=client) as connect:
            with self.assertRaisesRegex(ValueError, 'different model'):
                Store()
        client.create_collection.assert_not_called()
        client.close.assert_called_once()
        self.assertEqual(connect.call_args.kwargs['host'], '127.0.0.1')
        self.assertFalse(connect.call_args.kwargs['trust_env'])

    async def test_no_results_does_not_request_an_ungrounded_answer(self):
        import analysis
        with patch.object(cv_screener, 'query_qdrant', AsyncMock(return_value=[])), \
                patch.object(cv_screener.client.chat.completions, 'parse', AsyncMock()) as complete:
            answer = await analysis.analyze('Find a candidate', [])
        complete.assert_not_called()
        self.assertIn('не найдены', answer['content'])
        self.assertEqual(answer['sources'], [])
        self.assertIsNone(answer['analysis'])

    async def test_blank_query_is_rejected_before_search(self):
        with patch.object(cv_screener, 'query_qdrant', AsyncMock()) as query:
            with self.assertRaisesRegex(ValueError, 'Опишите вакансию'):
                await cv_screener.retrieve_sources('   ', [])
        query.assert_not_called()

    def test_sources_are_grouped_in_retrieval_order_without_duplicate_chunks(self):
        documents = [
            Doc(page_content='First fragment', metadata={'objectId': 'a', 'parent': 'a.txt'}),
            Doc(page_content='Other resume', metadata={'objectId': 'b', 'parent': 'b.txt'}),
            Doc(page_content='Second fragment', metadata={'objectId': 'a', 'parent': 'a.txt'}),
            Doc(page_content='First fragment', metadata={'objectId': 'a', 'parent': 'a.txt'}),
        ]
        self.assertEqual(cv_screener.group_sources(documents), [
            {'name': 'a.txt', 'text': 'First fragment\n\nSecond fragment'},
            {'name': 'b.txt', 'text': 'Other resume'},
        ])


if __name__ == '__main__':
    unittest.main()
