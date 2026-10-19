'''
Interface implementation for the Qdrant store
'''
from typing import List, Callable, cast, Dict
from uuid import uuid4
import sys

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import \
    PointStruct,  Filter, FieldCondition, Record, MatchValue, Range, SearchParams, \
    TextIndexParams, TokenizerType, PayloadSchemaType, TextIndexType, ScoredPoint
from qdrant_client.http import models
from qdrant_client.conversions import common_types as types

from common.schema import Doc, DocFilter, DocMetadata
from common.store import DocumentStoreBase
from common.config import Config

'''
NOTE: https://qdrant.tech/documentation/concepts/indexing/

This ***MAY*** be a problem, but since we provide our own embeddings, probably not...
However, this may affect the full text search and the parsing of words in the payload...

Qdrant does not support all languages out of the box:

multilingual - special type of tokenizer based on charabia package. It allows proper
tokenization and lemmatization for multiple languages, including those with non-latin
alphabets and non-space delimiters. See charabia documentation for full list of supported
languages supported normalization options. In the default build configuration, qdrant does
not include support for all languages, due to the increasing size of the resulting
binary. Chinese, Japanese and Korean languages are not enabled by default, but can be
enabled by building qdrant from source with
--features multiling-chinese,multiling-japanese,multiling-korean flags.
'''


class Store(DocumentStoreBase):
    _threshhold_search = .05
    client: AsyncQdrantClient
    payload_limit: int = 32000000
    batch_size: int = 500

    def __init__(self) -> None:
        '''
        Initializes the qdrant vector store. We use our own DocumentStore
        as a base here
        '''
        # Init the base
        super().__init__()

        # Get our configuation
        config = Config.getConfig()

        # Get the section name - should be qdrant
        sectionName = config['store']

        # Get our section
        section = config[sectionName]

        # Save our parameters
        self._collection = section['collection']
        self._similarity = section['similarity']
        self._renderChunkSize = section['renderChunkSize']
        self._host = '127.0.0.1' if section['host'] == 'localhost' else section['host']
        self._port = section['port']

        self.vectorSize = self.getVectorSize()

        # Init the store
        client = QdrantClient(
            host=self._host,
            port=self._port,
            trust_env=self._host not in ('localhost', '127.0.0.1', '::1'),
            prefer_grpc=False
        )

        if not client.collection_exists(self._collection):
            vector_params = types.VectorParams(
                size=self.getVectorSize(),
                distance=self._similarity)
            client.create_collection(
                collection_name=self._collection,
                vectors_config=vector_params)

        # See if we can get the collection info, throws if it does not exist or other error
        info = client.get_collection(
            collection_name=self._collection
        )

        vector_params = info.config.params.vectors
        if not isinstance(vector_params, models.VectorParams) or vector_params.size != self.vectorSize:
            client.close()
            raise ValueError('Your collection was created with a different model')

        # If we do not have a payload schema yet
        if not len(info.payload_schema):
            # Setup our payload index so we can query by nodeId
            client.create_payload_index(
                collection_name=self._collection,
                field_name='meta.nodeId',
                field_schema=PayloadSchemaType.KEYWORD
            )

            # Setup our payload index so we can query by object id
            client.create_payload_index(
                collection_name=self._collection,
                field_name='meta.objectId',
                field_schema=PayloadSchemaType.KEYWORD
            )

            # Setup our payload index so we can query by parent path
            client.create_payload_index(
                collection_name=self._collection,
                field_name='meta.parent',
                field_schema=PayloadSchemaType.KEYWORD
            )

            # Setup our payload index so we can query by permission id
            client.create_payload_index(
                collection_name=self._collection,
                field_name='meta.permissionId',
                field_schema=PayloadSchemaType.INTEGER
            )

            # Setup our payload index so we can query by isDeleted
            client.create_payload_index(
                collection_name=self._collection,
                field_name='meta.isDeleted',
                field_schema=PayloadSchemaType.BOOL
            )

            # Setup our payload index so we can query by isTable
            client.create_payload_index(
                collection_name=self._collection,
                field_name='meta.isTable',
                field_schema=PayloadSchemaType.BOOL
            )
            
            # Setup a full text keyword search on our content
            client.create_payload_index(
                collection_name=self._collection,
                field_name='content',
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                )
            )

        client.close()

        # Init the store
        self._client = AsyncQdrantClient(
            host=self._host,
            port=self._port,
            trust_env=self._host not in ('localhost', '127.0.0.1', '::1'),
            prefer_grpc=False
        )
        return

    def _convertFilter(
            self,
            *,
            docFilter: DocFilter) -> Filter:
        '''
        Build the generic filter based on required permissions, node, parent, etc
        '''
        # Declare the mist list to start addding conditions
        must: List[models.Condition] = []

        # If a nodeId was specified
        if docFilter.nodeId is not None:
            must.append(models.FieldCondition(
                key='meta.nodeId',
                match=models.MatchValue(value=docFilter.nodeId)
            ))

        if docFilter.isTable is not None:
            must.append(models.FieldCondition(
                key='meta.isTable',
                match=models.MatchValue(value=docFilter.isTable)
            ))

        if docFilter.tableIds is not None:
            must.append(models.FieldCondition(
                key='meta.tableId',
                match=models.MatchAny(any=docFilter.tableIds)
            ))

        # If a parent was specified
        if docFilter.parent is not None:
            must.append(models.FieldCondition(
                key='meta.parent',
                match=models.MatchText(text=docFilter.parent)
            ))

        # If a permissionId list was specified
        if docFilter.permissionId is not None:
            must.append(models.FieldCondition(
                key='meta.permissionId',
                match=models.MatchAny(any=docFilter.permissionId)
            ))

        # If a permissionId list was specified
        if docFilter.objectIds is not None:
            must.append(models.FieldCondition(
                key='meta.objectId',
                match=models.MatchAny(any=docFilter.objectIds)
            ))

        # If we are not going after deleted docs, add a condition
        if docFilter.isDeleted is None or not docFilter.isDeleted:
            must.append(models.FieldCondition(
                key='meta.isDeleted',
                match=models.MatchValue(value=False)
            ))

        # If we are not going after chunks, add a condition
        if docFilter.chunk is not None:
            must.append(models.FieldCondition(
                key='meta.chunk',
                match=models.MatchValue(value=docFilter.chunk)
            ))
        # Determine the basic must conditions
        return Filter(must=must)

    def _convertToDocs(self, points: List[ScoredPoint] | List[Record]) -> List[Doc]:
        '''
        Convert a list of points or records to a docGroup. Groups
        all document chunks  together
        '''
        docs: List[Doc] = []

        # Now, add the documents to the results
        for point in points:
            # If we don't have a payload, skip it
            if point.payload is None:
                continue

            # Get the payload content and meadata
            meta = cast(DocMetadata, point.payload['meta'])
            content = point.payload['content']

            # Determine the score of this document
            if isinstance(point, ScoredPoint):
                score = point.score
            else:
                score = 0

            # Create asearc new document
            doc = Doc(
                score=score,
                page_content=content,
                metadata=meta
            )

            # Append it to this documents chunks
            docs.append(doc)

        # Return it
        return docs

    async def acount_documents(self) -> int:
        '''
        Returns the number of vectors in the document store, not the number of
        documents themselves
        '''
        # Check it
        if self._client is None:
            raise Exception(
                'Client is not started')

        # Get the collection info
        info: types.CollectionInfo = await self._client.get_collection(
            collection_name=self._collection)

        return info.points_count or 0

    async def asearch(self,
                      query: str,
                      *,
                      docFilter: DocFilter) -> List[Doc]:
        '''
        Generic search - supports both keyword and semantic search with or
        without filters
        '''
        # Check it
        if self._client is None:
            raise Exception('Client is not started')

        # Declare the results list
        docs: List[Doc] = []

        # Build up the filter
        filter = self._convertFilter(docFilter=docFilter)

        # If we are doing a semantic search or not...
        if docFilter.useSemanticSearch:
            # Check the query string
            if query is None:
                raise BaseException('Query not specified')

            # We cannot support non-zero offsets
            if docFilter.offset:
                raise BaseException('Non-zero offset is not supported in semantic searching')

            # Embed the query
            embedding = self.encodeString(query)

            # Perform the search
            result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                query_filter=filter,
                with_vectors=False,
                with_payload=True,
                limit=docFilter.limit,
                score_threshold=self._threshhold_search,
                search_params=SearchParams(exact=True)
            )

            # Convert the points into groups
            docs = self._convertToDocs(result.points)
        else:
            # Add a condition for the keyword
            if filter.must is None:
                filter.must = []

            filter.must.append(models.FieldCondition(
                key='content',
                match=models.MatchText(text=query)
            ))

            # Perform the search
            records, nextURL = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=filter,
                offset=docFilter.offset,
                limit=docFilter.limit,
                with_vectors=False
            )

            # Convert the points into groups
            docs = self._convertToDocs(records)

        # Return them
        return docs

    async def aget(self, docFilter: DocFilter) -> List[Doc]:
        '''
        Given a filter, this will return the document groups matching the
        filter
        '''
        # Build up the filter
        filter = self._convertFilter(docFilter=docFilter)

        documents, offset = [], docFilter.offset or None
        while docFilter.limit is None or len(documents) < docFilter.limit:
            remaining = 500 if docFilter.limit is None else min(500, docFilter.limit - len(documents))
            records, offset = await self._client.scroll(
                collection_name=self._collection, scroll_filter=filter,
                offset=offset, limit=remaining, with_vectors=False)
            documents.extend(self._convertToDocs(records))
            if offset is None:
                break
        return documents

    async def agetPaths(self, parent: str | None = None, offset: int = 0, limit: int = 1000) -> Dict[str, str]:
        '''
        This will query and return all the unique parent paths
        '''

        # Build the base
        must: List[models.Condition] = [
            FieldCondition(
                key='meta.chunk',
                match=models.MatchValue(value=0)
            )
        ]

        # If parent specified, match on it
        if parent is not None:
            must.append(FieldCondition(
                key='meta.parent',
                match=models.MatchText(text=parent)
            ))

        # Build a filter to just ask for chunk 0
        filter = Filter(must=must)

        # Build up the path list
        paths: Dict[str, str] = {}

        # Perform the search
        records, nextPoint = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=filter,
            offset=offset,
            limit=limit,
            with_vectors=False,
            with_payload=True,
        )

        # Fill it in
        for record in records:
            # Get the payload
            payload = record.payload
            if payload is None:
                continue

            # Get the info
            meta = cast(DocMetadata, payload['meta'])

            # Get the parent
            parent = meta['parent']

            # Add it
            paths[parent] = meta['objectId']

        # And return what we found
        return paths

    async def aaddChunks(
            self,
            chunks: List[Doc]) -> None:
        '''
        Adds document chunks to the document store
        '''
        # Check it
        if self._client is None:
            raise Exception('Client is not started')

        # If no documents present, done
        if not len(chunks):
            return

        # Clear the points
        points: List[models.PointStruct] = []

        # Clear the object id list
        objectIds: Dict = {}

        async def flush():
            nonlocal points
            nonlocal objectIds
            ops = []

			# Build the batch operation for deletion
            if len(objectIds):
                ops.append(models.DeleteOperation(
                    delete=models.FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key='meta.objectId',
                                    match=models.MatchAny(
                                        any=list(objectIds.keys()))
                                    )
                                ]
                            )
                        )
                    )
                )

            # Build the batch operation for insert
            if len(points):
                ops.append(models.UpsertOperation(
                    upsert=models.PointsList(
                        points=points
                        )
                    )
                )

            # If we have nothing to do, done
            if not len(ops):
                return

            # Perform the batch
            await self._client.batch_update_points(
                collection_name=self._collection,
                update_operations=ops,
                wait=True
            )

            # Clear them
            objectIds = {}
            points = []

        # For each document
        for chunk in chunks:
            # Save this object id
            objectIds[chunk.metadata['objectId']] = True

        sum_size = 0
        cur_size = 0
        # For each document
        for chunk in chunks:
            # Get the embedding
            embedding = chunk.embedding

            # If we do not have an embedding
            if embedding is None:
                raise Exception('No embedding in document')

            # Append the points
            tmp_struct = PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        'content': chunk.page_content,
                        'meta': chunk.metadata
                    }
                )
            cur_size = sys.getsizeof(tmp_struct)
            sum_size += cur_size
            points.append(tmp_struct)

            # If there are more than 
            if (len(points) >= self.batch_size) or (sum_size + cur_size > self.payload_limit):
                await flush()
                cur_size = 0
                sum_size = 0
        
        # Flush any stragglers
        await flush()
        return

    async def aremove(
            self,
            objectIds: List[str]) -> None:
        '''
        Deletes all documents with a matching objectIds from the document store.
        '''
        # Check it
        if self._client is None:
            raise Exception('Client is not started')

        # Build a filter for an object id
        filter_objectId = Filter(
            must=[
                FieldCondition(
                    key='meta.objectId',
                    match=models.MatchAny(any=objectIds)
                )
            ]
        )

        # Delete the points with the given object Id
        await self._client.delete(
            collection_name=self._collection,
            points_selector=filter_objectId,
            wait=True

        )
        return

    async def amarkDeleted(
            self,
            objectIds: List[str]) -> None:
        '''
        Marks the set of documents with the given objectId as deleted. They
        then will not be returned from the search without specifying deleted=True
        '''
        # Check it
        if self._client is None:
            raise Exception('Client is not started')

        # Build a filter for an object id
        filter_objectId = Filter(
            must=[
                FieldCondition(
                    key='meta.objectId',
                    match=models.MatchAny(any=objectIds)
                )
            ]
        )

        # Set all the objects with the given objectId to true
        await self._client.set_payload(
            collection_name=self._collection,
            payload={
                'isDeleted': True
            },
            points=filter_objectId
        )
        return

    async def amarkActive(
            self,
            objectIds: List[str]) -> None:
        '''
        Marks the set of documents with the given objectId as active. This occurs
        if a document now 'comes back' after begin deleted
        '''
        # Check it
        if self._client is None:
            raise Exception('Client is not started')

        # Build a filter for an object id
        filter_objectId = Filter(
            must=[
                FieldCondition(
                    key='meta.objectId',
                    match=models.MatchAny(any=objectIds)
                )
            ]
        )

        # Set all the objects with the given objectId to false
        await self._client.set_payload(
            collection_name=self._collection,
            payload={
                'isDeleted': False
            },
            points=filter_objectId
        )

    async def arender(
            self,
            objectId: str,
            callback: Callable[[str], None]) -> None:
        '''
        Given an object id, render the complete document. Rehydrates all the
        chunks into the proper order.
        '''
        # Check it
        if self._client is None:
            raise Exception('Client is not started')

        # Since chunks are returned in any order, and a single objectId
        # may contain tens of thousands of chunks, we grave them one
        # group at a time (renderChunkSize), put them into an array,
        # join them and call the callback
        offset = 0
        while (True):
            # Build  filter for getting a set of chunks
            filter_range = FieldCondition(
                key='meta.chunk',
                range=Range(
                    gte=offset, lt=offset + self._renderChunkSize)
            )

            # Build a filter for an object id
            filter_objectId = FieldCondition(
                key='meta.objectId',
                match=MatchValue(value=objectId)
            )

            # Build a filter to include only content - no tables
            filter_noTables = FieldCondition(
                key='meta.isTable',
                match=MatchValue(value=False)
            )

            # Perform the query
            records, nextPoint = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[filter_objectId, filter_range, filter_noTables]
                ),
                limit=self._renderChunkSize,
                with_payload=True
            )

            # Create a renderChunkSize array with empty
            # entries. This will allow us to join even when
            # a chunk doesn't come back
            text: List[str] = [''] * self._renderChunkSize
            lastIndex = -1

            # Now, add the documents to the results
            for record in records:
                # Get the payload
                payload = record.payload
                if payload is None:
                    continue

                # Get the info
                meta = cast(DocMetadata, payload['meta'])
                content = payload['content']
                chunk = meta['chunk']

                # Should never happen since we gave it an offset
                if chunk < offset:
                    continue

                # Should never happen since we gave it a range
                if chunk >= offset + self._renderChunkSize:
                    continue

                # Get the index into the array
                index = chunk - offset

                # Add it to our array
                text[index] = content

                # Determine the highest index we use
                if index > lastIndex:
                    lastIndex = index

            # Compute the number of items we are going to process
            numberOfItems = lastIndex + 1

            # If we got no items back, we are done
            if numberOfItems < 1:
                break

            # Join it together
            fullText = ''.join(text[0:numberOfItems])

            # Call the output function
            callback(fullText)

            # If we got less than we asked for, must be done
            if (numberOfItems < self._renderChunkSize):
                break
