'''
Generates context information for both semantic search and keyword search
'''
from typing import List
from common.schema import Doc


class Context:
    '''
    This class creates context information around a document
    '''

    def __init__(self):
        '''
        Create the context generator
        '''

    def __call__(
        self,
        query: str,
        useSemanticSearch: bool,
        documents: List[Doc]
    ) -> List[Doc]:
        '''
        This uses a model to perform context marking within the document.
        Can be either keyword or semantic search results.
        '''
        # Iterate through the groups
        for doc in documents:
            # Setup the context
            doc.context = [doc.page_content]

        # Return the modified document groups
        return documents
