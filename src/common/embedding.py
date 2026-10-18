'''
This is the common embedding module
'''
import importlib
import warnings
from abc import abstractmethod, ABC
from typing import List

from common.config import Config
from common.schema import Doc


class EmbeddingBase(ABC):
    '''
    The embedding base allows the actual embedding internals to be extracted
    into provider implementations. They will usually be a SentenceTransformet,
    a Transformer or Gpt4All
    '''
    @abstractmethod
    def getVectorSize(self) -> int:
        '''
        Returns the vector size of the embedding module
        '''

    @abstractmethod
    def getMaximumTokens(self) -> int:
        '''
        Returns the maximum number of tokens in a request
        '''

    @abstractmethod
    def encodeString(self, string: str) -> List:
        '''
        Encode the query string into a vector
        '''

    @abstractmethod
    def encodeChunks(self, chunks: List[Doc]) -> None:
        '''
        Add the embedding member to each of the listed document
        chunks under doc.embedding
        '''

    @abstractmethod
    def getTokens(self, text: str) -> int:
        '''
        Gets the number of tokens required using the model
        '''


def getEmbedding() -> EmbeddingBase:
    '''
    Looks at the configuration and returns and initializes an embedding
    '''
    # Get the configuration
    config = Config.getConfig()

    # Pull out the store we are going to use
    store = config['embedding']

    # Get the provider name
    provider = config[store]['provider']

    # Build up the module name - it will be in the store dir
    name = 'providers.embedding.' + provider

    # Get the module
    module = importlib.import_module(name)

    # This is a warning from pytorch that the sentence transformers
    # use. We really can't do anything about it at this time except
    # wait for transformers to be updated
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")

    # Create the embedding
    return getattr(module, 'Embedding')()
