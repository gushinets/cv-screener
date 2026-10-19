'''
This is the common embedding module
'''
from typing import List
from numpy import ndarray

from sentence_transformers import SentenceTransformer  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from common.config import Config
from common.embedding import EmbeddingBase
from common.schema import Doc


class Embedding(EmbeddingBase):
    '''
    The embedding class controls the conversion of a piece of text
    into a vector. This can be a query or a set of document chunks.
    '''

    '''
    Privates
    '''
    _model: str = ''
    _modelFolder: str = ''
    _vectorSize: int = 0
    _tokenSize: int = 0
    _embedding: SentenceTransformer = None
    _tokenCounter: AutoTokenizer = None

    def __init__(self):
        # Init the base
        super().__init__()

        # Get our configuation
        config = Config.getConfig()

        # Get the section name - should be qdrant
        sectionName = config['embedding']

        # Get our section
        section = config[sectionName]
        self._batchSize = section.get('batch_size', 32)
        if 'cpu_threads' in section:
            import torch
            torch.set_num_threads(section['cpu_threads'])

        # Get the model from our section
        self._model = section['model']

        # Get the max tokens
        self._tokenSize = section['tokens']

        # Get the model folder
        self._modelFolder = Config.getModelCacheFolder()

        # Create the embedding
        self._embedding = SentenceTransformer(
            model_name_or_path=self._model,
            cache_folder=self._modelFolder,
            revision=section.get('revision')
        )

        # Check it
        if self._embedding is None:
            raise Exception('No embedding')

        # Get the vetor size
        vectorSize = self._embedding.get_embedding_dimension()
        if vectorSize is None:
            raise Exception('Unable to determine vector size')

        # Save it
        self._vectorSize = vectorSize

        # Get a token counter
        # Note the different variable names for the cache dir from above
        self._tokenCounter = AutoTokenizer.from_pretrained(
            self._model,
            cache_dir=self._modelFolder,
            revision=section.get('revision'))

    def getVectorSize(self) -> int:
        '''
        Returns the vector size of the embedding module
        '''
        # Return it
        return self._vectorSize

    def getMaximumTokens(self) -> int:
        '''
        Returns the maximum number of tokens in a request
        '''
        # Return it
        return self._tokenSize

    # Encode the given string return a list of vectors
    def encodeString(self, string: str) -> List:
        '''
        Given a string (document), encode the document into a
        vector and return the vector as an array of floats
        '''
        # Get the vectors
        vectors = self._embedding.encode(string, show_progress_bar=False)

        # Check the return type
        if not isinstance(vectors, ndarray):
            raise Exception('Embedding is not an ndarray')

        # Encode it
        return vectors.tolist()

    # Encode the given string return a list of vectors
    def encodeChunks(self, chunks: List[Doc]) -> None:
        '''
        Given a list of documents, encode the document into a
        vector and place it in the embedding field of the document
        '''
        # For each document, if specified
        embeddings: List[str] = []
        for chunk in chunks:
            embeddings.append(chunk.page_content)

        # Get the vectors
        array_vectors = self._embedding.encode(
            embeddings, show_progress_bar=False, batch_size=self._batchSize)

        # Save the embeddings
        for index in range(0, len(chunks)):
            # Save the embedding as a list
            chunks[index].embedding = array_vectors[index].tolist()
            index = index + 1

    def getTokens(self, text: str) -> int:
        '''
        This function will determine how many tokens, according to the model
        that the given string will take
        '''
        # Return the token count
        #
        # Note - we turn verbosity off here since we get a warning if we
        # pass over a string which results in too many tokens. The splitter
        # does this specifically to find sequences that are too long, then
        # splits those up...
        return len(self._tokenCounter.encode(text, verbose=False))
