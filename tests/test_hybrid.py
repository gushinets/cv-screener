import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.schema import Doc
from common.hybrid import rank_documents, tokenize


def chunk(name, text, number=0, score=0):
    return Doc(page_content=text, score=score,
               metadata={'parent': name, 'objectId': name, 'chunk': number})


class HybridTests(unittest.TestCase):
    def test_independent_bm25_adds_a_document_missing_from_dense_pool(self):
        dense = chunk('A.txt', 'Python backend', score=.9)
        lexical = chunk('B.txt', 'quasar42 database')
        indexed = [dense, lexical, chunk('C.txt', 'Java service')]
        result = rank_documents('quasar42', [dense], indexed)
        self.assertEqual([d.metadata['parent'] for d in result], ['A.txt', 'B.txt'])
        self.assertIs(result[1], lexical)

    def test_repeated_chunks_and_exact_resume_copies_do_not_get_extra_votes(self):
        first = chunk('A.txt', 'Avery Reed\nPython', score=.9)
        tail = chunk('A.txt', 'PostgreSQL', 1, .8)
        copy = chunk('Copy.txt', first.page_content, score=.7)
        copy_tail = chunk('Copy.txt', tail.page_content, 1)
        other = chunk('B.txt', 'Morgan Reed\nJava', score=.6)
        result = rank_documents('unmatched', [first, tail, copy, other], [copy_tail, other, copy, tail, first])
        self.assertEqual([d.metadata['parent'] for d in result], ['A.txt', 'B.txt'])

    def test_zero_lexical_scores_preserve_dense_order_and_do_not_mutate_scores(self):
        first, second = chunk('A.txt', 'Java', score=.8), chunk('B.txt', 'Python', score=.9)
        result = rank_documents('unmatched', [second, first], [first, second])
        self.assertEqual(result, [second, first])
        self.assertEqual([first.score, second.score], [.8, .9])

    def test_explicit_filename_survives_rank_out_and_limit(self):
        dense = [chunk(f'noise-{n}.txt', f'Candidate{n} Python', score=1-n/100) for n in range(30)]
        named = chunk('Named.docx', 'Morgan Reed\nQuasar42')
        result = rank_documents('Сравни NAMED.DOCX и Python', dense, [*dense, named])
        self.assertEqual(len(result), 25)
        self.assertIs(result[0], named)

    def test_explicit_duplicate_filename_opens_the_requested_copy(self):
        original = chunk('A.txt', 'Avery Reed\nPython', score=.9)
        copy = chunk('Copy.txt', original.page_content)
        result = rank_documents('Copy.txt', [original], [original, copy])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], copy)

    def test_empty_index_and_unindexed_dense_results_do_not_leak_sources(self):
        self.assertEqual(rank_documents('Python', [], []), [])
        orphan = chunk('orphan.txt', 'Private unindexed resume', score=.9)
        indexed = chunk('Indexed.txt', 'Java')
        self.assertNotIn(orphan, rank_documents('unmatched', [orphan], [indexed]))

    def test_technical_names_and_slash_lists(self):
        self.assertEqual(tokenize('C++ C# .NET, Python/Playwright/pytest UI/API CI/CD Pytest-xdist Ёлка'),
                         ['c++', 'c#', '.net', 'python', 'playwright', 'pytest', 'ui', 'api', 'ci', 'cd',
                          'pytest-xdist', 'елка'])


if __name__ == '__main__':
    unittest.main()
