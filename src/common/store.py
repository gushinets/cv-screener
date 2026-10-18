'''
This is the base class for the document store. It loads the appropriate module
from the store directory based on store selected
'''
import asyncio
from common.config import Config
import importlib
from abc import abstractmethod, ABC
from typing import List, Callable, Dict
from common.embedding import getEmbedding, EmbeddingBase
from common.preprocessor import PreProcessor
from common.schema import Doc, DocFilter, DocMetadata



class DocumentStoreBase(ABC):
    '''
    The DocumentStoreBase class is used to abstract the details of the DocumentStore
    so it can be dynamically changed to a different provider. Some methods are abstract
    which must be implemented in the actual providers, but some of the utility functions
    are implemented here but can be overridden
    '''

    '''
    Privates
    '''
    _embedding: EmbeddingBase
    _preprocessor: PreProcessor

    def __init__(self):
        '''
        Initializes the store. The __init__ constructor is not part of the Store Protocol
        and the signature can be customized to your needs. For example, parameters needed
        to set up a database client would be passed to this method.
        '''
        # Add it to this instance
        self._embedding = getEmbedding()
        self._preprocessor = PreProcessor(embedding=self._embedding)

    @abstractmethod
    async def acount_documents(self) -> int:
        '''
        Returns how many documents are present in the document store.
        '''

    def count_documents(self) -> int:
        '''
        Returns how many documents are present in the document store.
        '''
        return asyncio.run(self.acount_documents())

    @abstractmethod
    async def asearch(self,
                      query: str,
                      *,
                      docFilter: DocFilter) -> List[Doc]:
        '''
        Returns the documents that match the filters provided. If useSemanticSearch is True,
        a query string or embedding needs to be provided. If a query string is provided,
        it will be used rather than the embedding. If useSemanticSearch is False, the query
        string contains the keyword to search for.
        '''

    async def search(self,
                     query: str,
                     *,
                     docFilter: DocFilter) -> List[Doc]:
        '''
        Returns the documents that match the filters provided. If useSemanticSearch is True,
        a query string or embedding needs to be provided. If a query string is provided,
        it will be used rather than the embedding. If useSemanticSearch is False, the query
        string contains the keyword to search for.
        '''
        return asyncio.run(self.asearch(
            query=query,
            docFilter=docFilter
        ))

    @abstractmethod
    async def aget(self, docFilter: DocFilter) -> List[Doc]:
        '''
        Performs a database query to get objects
        '''

    def get(self, docFilter: DocFilter) -> List[Doc]:
        '''
        Performs a database query to get objects
        '''
        return asyncio.run(self.aget(docFilter))

    @abstractmethod
    async def agetPaths(self, parent: str | None = None, offset: int = 0, limit: int = 1000) -> Dict[str, str]:
        '''
        This will query and return all the unique parent paths
        '''

    def getPaths(self, parent: str | None = None, offset: int = 0, limit: int = 1000) -> Dict[str, str]:
        '''
        This will query and return all the unique parent paths
        '''
        return asyncio.run(self.agetPaths(parent, offset, limit))

    @abstractmethod
    async def aaddChunks(self, chunks: List[Doc]) -> None:
        '''
        Writes (or overwrites) documents into the store.
        '''

    def addChunks(self, chunks: List[Doc]) -> None:
        '''
        Writes (or overwrites) documents into the store.
        '''
        return asyncio.run(self.aaddChunks(chunks))

    @abstractmethod
    async def aremove(self, objectIds: List[str]) -> None:
        '''
        Deletes all documents with a matching objectIds from the document store
        '''

    def remove(self, objectIds: List[str]) -> None:
        '''
        Deletes all documents with a matching objectIds from the document store
        '''
        return asyncio.run(self.aremove(objectIds))

    @abstractmethod
    async def amarkDeleted(self, objectIds: List[str]) -> None:
        '''
        Marks the set of documents with the given objectId as deleted. They
        then will not be returned from the search without specifying deleted=True
        '''

    def markDeleted(self, objectIds: List[str]) -> None:
        '''
        Marks the set of documents with the given objectId as deleted. They
        then will not be returned from the search without specifying deleted=True
        '''
        return asyncio.run(self.amarkDeleted(objectIds))

    @abstractmethod
    async def amarkActive(self, objectIds: List[str]) -> None:
        '''
        Marks the set of documents with the given objectId as active. This occurs
        if a document now "comes back" after begin deleted
        '''

    def markActive(self, objectIds: List[str]) -> None:
        '''
        Marks the set of documents with the given objectId as active. This occurs
        if a document now "comes back" after begin deleted
        '''
        return asyncio.run(self.amarkActive(objectIds))

    @abstractmethod
    async def arender(self, objectId: str, callback: Callable[[str], None]) -> None:
        """
        Given an object id, render the complete document. Rehydrates all the
        chunks into the proper order.
        """

    def render(self, objectId: str, callback: Callable[[str], None]) -> None:
        """
        Given an object id, render the complete document. Rehydrates all the
        chunks into the proper order.
        """
        return asyncio.run(self.arender(objectId, callback))

    #
    # The following can be overriden, but not necessary as the default functionality
    # is generic and will work across multiple vector store providers
    #
    def getVectorSize(self) -> int:
        '''
        Return the size of the embedding that was selected
        '''
        return self._embedding.getVectorSize()

    def preprocess(self, document: Doc) -> List[Doc]:
        '''
        PreProcess a document into a series of chunks
        '''
        # Check it
        if self._preprocessor is None:
            raise Exception('Preprocessor not created')

        # Process the document
        return self._preprocessor.process(document)

    def encodeString(self, query: str) -> List:
        '''
        Create an embedding for a query string
        '''
        # Check it
        if self._embedding is None:
            raise Exception('Embedding not created')

        # Encode it
        return self._embedding.encodeString(query)

    def encodeDocuments(self, documents: List[Doc]) -> None:
        '''
        Create an embedding for a query string
        '''
        # Check it
        if self._embedding is None:
            raise Exception('Embedding not created')

        # Encode them
        return self._embedding.encodeChunks(documents)

    async def aadd(
            self,
            objectId: str,
            nodeId: str,
            parent: str,
            permissionId: int,
            text: str,
            tables: str) -> None:
        '''
        Adds a document to to the store. Processes the docments and
        tables into checks, computes the embeddings, etc.
        '''
        # Check it
        if self._embedding is None:
            raise Exception('Embedding not created')

        # Create the metadata
        metadata: DocMetadata = {
            'objectId': objectId,
            'chunk': 0,
            'nodeId': nodeId,
            'parent': parent,
            'permissionId': permissionId,
            'isTable': False,
            'isDeleted': False,
            'tableId': 0,
        }

        # Build up a document object to pass to the preprocessor
        document = Doc(
            page_content=text,
            page_tables=tables,
            metadata=metadata
        )

        # Retrieve it's chunks
        chunks = await asyncio.to_thread(self.preprocess, document)

        # if no chunks there is nothing to process
        if not chunks:
            raise ValueError('Document contains no indexable text')

        # Create the embeddings
        await asyncio.to_thread(self.encodeDocuments, chunks)

        # Add the chunks
        await self.aaddChunks(chunks)

    def add(
            self,
            objectId: str,
            nodeId: str,
            parent: str,
            permissionId: int,
            text: str,
            tables: str) -> None:
        '''
        Adds a document to to the store. Processes the docments and
        tables into checks, computes the embeddings, etc.
        '''
        return asyncio.run(self.aadd(
            objectId,
            nodeId,
            parent,
            permissionId,
            text,
            tables
        ))


def getStore() -> DocumentStoreBase:
    '''
    Looks at the configuration and returns and initializes a Store
    '''
    # Get the configuration
    config = Config.getConfig()

    # Pull out the store we are going to use
    store = config['store']

    # Get the provider name
    provider = config[store]['provider']

    # Build up the module name - it will be in the store dir
    name = 'providers.store.' + provider

    # Get the module
    module = importlib.import_module(name)

    # Instantiate the store class
    return getattr(module, 'Store')()
