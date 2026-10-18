'''
Search class - implements the search interface
'''
import asyncio
from typing import List, Optional, cast
from common.schema import Doc, DocFilter
from common.store import DocumentStoreBase, getStore
from common.context import Context
from common.rerank import ReRank


class Search():
    '''
    Provides the main interface for search. Once instantiated, use the search() method
    to search for documents within document store
    '''

    '''
    Privates
    '''
    _store: DocumentStoreBase
    _context: Context

    def __init__(self) -> None:
        '''
        Create the search pipeline
        '''
        self._store = getStore()
        self._context = Context()

    async def _processTables(self,
                             docs: List[Doc]) -> List[Doc]:
        '''
        Process tables. This is post processing after a search has
        completed which reads the entire table anc concatenates
        the documents of the tables into the first entry for
        that table
        '''
        chunks: List[Doc] = []
        tables: List[str] = []

        # Walk through all the groups
        for doc in docs:
            # If this is not a table, put it into the text doc list
            if not doc.metadata['isTable']:
                chunks.append(doc)
                continue

            # Get the table id
            objectId = doc.metadata['objectId']
            tableId = cast(int, doc.metadata['tableId'])

            # Get a key into the table of this table
            key = objectId + '.' + str(tableId)

            # If we have already read this table, skip it
            if key in tables:
                continue

            # Get a filter to read only the table specified
            tableFilter = DocFilter({
                "isTable": True,
                "objectIds": [objectId],
                "tableIds": [tableId]
            })

            # Get all the table chunks for this table in the document
            tableChunks = await self._store.aget(tableFilter)

            # Sort them by chunk
            tableChunks.sort(key=lambda doc: doc.metadata['chunk'])

            # Gather up all the text of the table
            tableText = ''
            for chunk in tableChunks:
                # Append the text
                tableText = tableText + chunk.page_content

            # Save the reconstitued table in the document
            doc.page_content = tableText

            # Save the fact that we processed this table so if we see
            # another table reference further down, we just skip it
            tables.append(key)

            # Append this full table chunk to the list
            chunks.append(doc)

        # Return the groups
        return chunks

    def getStore(self):
        '''
        Returns the underlying store so it can be reused
        '''
        return self._store

    async def asearch(self,
                      query: str,
                      *,
                      docFilter: Optional[DocFilter] = None) -> List[Doc]:
        '''
        Executes a query with the given query and filters. If useSemanticSearch is set to True, then
        query contains a string to vectorize and search for. If False, query is considered a
        filter which will convert over to a keyword query without a vector.
        '''
        # Get a filter if one is not supplied
        if docFilter is None:
            docFilter = DocFilter()

        # Check it
        if self._store is None:
            raise Exception('Store not instantiated')

        # Run a query
        docs = await self._store.asearch(
            query=query,
            docFilter=docFilter
        )

        # Process tables
        if len(docs) and docFilter.useTableConcat:
            docs = await self._processTables(docs)

        # Rerank the groups and all its documents
        if len(docs) and docFilter.useQuickRank:
            docs = ReRank.rerankDocuments(query=query, documents=docs)

        # If we need context, do so now
        if docFilter.context is None or docFilter.context:
            docs = self._context(query=query, useSemanticSearch=docFilter.useSemanticSearch, documents=docs)

        # Return the document set
        return docs

    def search(self,
               query: str,
               *,
               docFilter: Optional[DocFilter] = None) -> List[Doc]:
        '''
        Executes a query with the given query and filters. If useSemanticSearch is set to True, then
        query contains a string to vectorize and search for. If False, query is considered a
        filter which will convert over to a keyword query without a vector.
        '''
        return asyncio.run(self.asearch(
            query=query,
            docFilter=docFilter
        ))
