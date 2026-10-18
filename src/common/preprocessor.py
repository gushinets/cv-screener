'''
This is the common preprocess for documents that break a large document
into chunks
'''
import copy

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import Config
from common.schema import Doc
from common.embedding import EmbeddingBase


class PreProcessor:
    '''
    The preprocessor class cleans and splits text
    '''

    '''
    Privates
    '''
    _preprocessor: RecursiveCharacterTextSplitter
    _embedding: EmbeddingBase

    def __init__(self, embedding: EmbeddingBase):
        # Init the base
        super().__init__()

        # Get the configuration
        config = Config.getConfig()

        # Get the parameters
        param = config['preprocessor']

        # Get the embedding to we can tokenize properly
        self._embedding = embedding

        # Build the preprocessor
        self._preprocessor = RecursiveCharacterTextSplitter(
            length_function=self._embedding.getTokens,
            **param)

    # Process a document by splitting it into chunks
    def process(self, document: Doc) -> List[Doc]:
        # Build the entire array
        result = []
        chunkId = 0

        # --- Process document page_content first
        chunks = self._preprocessor.split_documents([document])

        # For each document that was returned (split)
        for index in range(0, len(chunks)):
            # Add the chunk id to the metadata
            chunks[index].metadata['isTable'] = False
            chunks[index].metadata['chunk'] = chunkId

            # And add it to the result
            result.append(Doc(
                page_content=chunks[index].page_content,
                metadata=chunks[index].metadata,
            ))

            # Next chunk
            chunkId = chunkId + 1

        # --- Process documet page_tables next
        tableText = document.page_tables
        if tableText:
            # Split up the tables leaving the end tag intact
            tables = tableText.split('***\n')

            # For each table
            for tableId, table in enumerate(tables):
                # Split the table into rows
                splitText = table.split('\n')

                # Figure out how many tokens each row has
                lineTokens: List[int] = []
                lineText: List[str] = []
                for line in splitText:
                    # Throw out blank lines
                    line = line.strip()
                    if not line:
                        continue

                    # Get the embedding
                    tokens = self._embedding.getTokens(line.strip())
                    lineText.append(line)
                    lineTokens.append(tokens)

                # Now build up the strings
                currentText = ''
                currentTokens = 0

                # Get the maximum number of tokens per request and leave
                # a little bit left over
                maxTokens = self._embedding.getMaximumTokens() - 16

                # Loop through the rows adding until we hit max tokens. By
                # going one more than the last line, we can force the this
                # to output the remaining
                for index in range(0, len(lineText) + 1):
                    # If this does not overflow our token count...
                    if index < len(lineText) and currentTokens + lineTokens[index] < maxTokens:
                        currentText = currentText + lineText[index] + '\n'
                        currentTokens = currentTokens + lineTokens[index]
                        continue

                    # If we don't have any text, skip it. Happens if the
                    # last line fit in the last chunk
                    if not currentText:
                        continue

                    # Copy the metadata
                    metadata = copy.deepcopy(document.metadata)

                    # Create a document of just the table text
                    tableDocument = Doc(
                        page_content=currentText,
                        metadata=metadata
                    )

                    # Show this as a table item and save the chunk id
                    tableDocument.metadata['isTable'] = True
                    tableDocument.metadata['tableId'] = tableId
                    tableDocument.metadata['chunk'] = chunkId

                    # Append it to the list
                    result.append(tableDocument)

                    # Next chunk
                    chunkId = chunkId + 1

                    # Save the line that caused the overflow
                    if index < len(lineText):
                        currentText = lineText[index] + '\n'
                        currentTokens = lineTokens[index]

        # Return the results
        return result
