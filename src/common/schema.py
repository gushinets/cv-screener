'''
Defines out document format which is derived from the
langchain document
'''
import os
from langchain_core.documents import Document
from typing import List, TypedDict, Optional, Dict, Any


class DocFilter:
    '''
    The doc filter class contains information that controls
    the search, the question and the conversational interfaces
    '''

    # Semantic search true - use vectors, otherwise use keyword search
    useSemanticSearch: bool = True

    # Full tables are read and concatenated together
    useTableConcat: Optional[bool] = None

    # Perform a quick rerank
    useQuickRank: bool = False

    # Perform a group rerank
    useGroupRank: bool = True

    # For questions, number of follow up questions
    followUpQuestions: int = 5

    offset: int = 0
    limit: int | None = 25

    # Include context in the search results
    context: bool = False

    # Include content in the search results
    content: bool = True

    # Document search filters
    nodeId: Optional[str] = None
    parent: Optional[str] = None
    name: Optional[str] = None
    permissionId: Optional[List[int]] = None
    isDeleted: Optional[bool] = None
    objectIds: Optional[List[str]] = None
    chunk: Optional[int] = None
    isTable: Optional[bool] = None
    tableIds: Optional[List[int]] = None

    def __init__(self, init: Dict[str, Any] = {}) -> None:
        # For all the keys specified
        for key in init:
            # If it is a valid get
            if hasattr(self, key):
                # Set the value
                setattr(self, key, init[key])


class DocMetadata(TypedDict):
    '''
    Represents the metadata stored with each chunk of a document
    '''
    objectId: str
    chunk: int
    nodeId: str
    parent: str
    permissionId: int
    isDeleted: bool
    isTable: bool
    tableId: int


class DocDict(DocMetadata):
    '''
    Represents a document as a dict
    '''
    score: float
    context: Optional[List[str]]
    content: Optional[str]


class Doc(Document):
    '''
    Represents a document check derived from Document as a class
    '''
    page_tables: str | None = None
    embedding: List[float] = []
    score: float = 0.0
    context: List[str] | None = []
    tokens: int | None = None

    def __init__(
            self,
            embedding: List[float] | None = None,
            score: float = 0.0,
            page_content: str = '',
            **kwargs):
        super().__init__(page_content=page_content, **kwargs)
        self.embedding = embedding if embedding is not None else []
        self.score = score
        self.tokens = None
        self.context = None

    def __repr__(self):
        filename = os.path.basename(self.metadata['parent'])
        return f"{filename}/{self.metadata['chunk']}={self.score}"

    def toDict(self, content: bool = True, context: bool = False) -> DocDict:
        # Build up the basic document
        doc: DocDict = {
            # Build up our result
            'objectId': self.metadata['objectId'],
            'chunk': self.metadata['chunk'],
            'nodeId': self.metadata['nodeId'],
            'parent': self.metadata['parent'],
            'permissionId': self.metadata['permissionId'],
            'isDeleted': self.metadata['isDeleted'],
            'isTable': self.metadata['isTable'],
            'tableId': -1 if not self.metadata['isTable'] else self.metadata['tableId'],
            'score': self.score,
            'content': None,
            'context': None
        }

        # If caller wants content, set it
        if content:
            doc['content'] = self.page_content

        # If caller wants context, set it
        if context:
            doc['context'] = self.context

        # Return the dict
        return doc


class DocGroupDict(TypedDict):
    '''
    Represents a document group as a dict
    '''
    objectId: str
    parent: str
    score: float
    documents: List[DocDict]


class DocGroup:
    '''
    Represents a collection of chunks from a single document
    '''

    def __init__(
            self,
            score: float = 0.0,
            objectId: str = '',
            parent: str = '',
            documents: List[Doc] | None = None):
        self.score = score
        self.objectId = objectId
        self.parent = parent
        self.documents = documents if documents is not None else []

    def __repr__(self):
        filename = os.path.basename(self.parent)
        return f"{filename}={self.score}"

    def toDict(self, content: bool = True, context: bool = False):
        # Setup the basic group
        group: DocGroupDict = {
            'objectId': self.objectId,
            'parent': self.parent,
            'score': self.score,
            'documents': []
        }

        # Convert all the documents over
        for doc in self.documents:
            group['documents'].append(doc.toDict(content, context))

        # And return this group
        return group


class DocReference(TypedDict):
    objectId: str
    document: str


class DocEntity(TypedDict):
    entity: str
    type: str
