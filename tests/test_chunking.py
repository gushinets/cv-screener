import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.config import Config
from common.preprocessor import PreProcessor
from common.schema import Doc


class ChunkingTests(unittest.TestCase):
    def test_bge_indexing_uses_the_configured_small_batch(self):
        import numpy as np
        from providers.embedding.sentenceTransformer import Embedding
        embedding = Embedding.__new__(Embedding)
        embedding._batchSize = 8
        embedding._embedding = Mock()
        embedding._embedding.encode.return_value = np.array([[.1, .2], [.3, .4]])
        documents = [Doc(page_content='Avery Python'), Doc(page_content='Morgan Java')]
        embedding.encodeChunks(documents)
        embedding._embedding.encode.assert_called_once_with(
            ['Avery Python', 'Morgan Java'], show_progress_bar=False, batch_size=8)
        self.assertEqual(documents[1].embedding, [.3, .4])

    def test_bge_cpu_threads_follow_the_measured_profile(self):
        from providers.embedding.sentenceTransformer import Embedding
        config = {'embedding': 'bgeM3', 'bgeM3': {'model': 'test-bge', 'tokens': 256, 'cpu_threads': 6}}
        with patch.object(Config, 'getConfig', return_value=config), \
                patch('providers.embedding.sentenceTransformer.SentenceTransformer') as model, \
                patch('providers.embedding.sentenceTransformer.AutoTokenizer'), \
                patch('torch.set_num_threads') as threads:
            model.return_value.get_embedding_dimension.return_value = 1024
            Embedding()
        threads.assert_called_once_with(6)

    def test_shared_length_counters_bound_chunks_and_keep_configuration_unchanged(self):
        params = {'chunk_size': 8, 'chunk_overlap': 0,
                  'length_tokenizers': [{'model': 'test-counter', 'prefix': 'passage: '}]}
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text.split())*2)))
        embedding = SimpleNamespace(getTokens=lambda text: len(text.split()))
        with patch.object(Config, 'getConfig', return_value={'preprocessor': params}), \
                patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer) as load:
            processor = PreProcessor(embedding)
            chunks = processor.process(Doc(page_content='alpha beta gamma delta epsilon zeta eta theta'))
        self.assertTrue(chunks)
        self.assertTrue(all(len(tokenizer.encode('passage: '+c.page_content)) <= 8 for c in chunks))
        self.assertEqual(' '.join(c.page_content for c in chunks), 'alpha beta gamma delta epsilon zeta eta theta')
        self.assertEqual(params['length_tokenizers'][0]['prefix'], 'passage: ')
        load.assert_called_once()

    def test_embedding_only_configuration_still_works(self):
        params = {'chunk_size': 3, 'chunk_overlap': 0}
        embedding = SimpleNamespace(getTokens=lambda text: len(text.split()))
        with patch.object(Config, 'getConfig', return_value={'preprocessor': params}):
            processor = PreProcessor(embedding)
            chunks = processor.process(Doc(page_content='alpha beta gamma delta'))
        self.assertTrue(all(embedding.getTokens(c.page_content) <= 3 for c in chunks))


if __name__ == '__main__':
    unittest.main()
