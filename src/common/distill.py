'''
Distill a set of documents down to process the given prompt
'''
import json
import asyncio
from typing import List, TypedDict, cast

from .prompts import Prompts
from ..common.chat import ChatBase
from ..common.util import safeString, parseJson
from ..common.schema import Doc, DocFilter, DocGroup, DocReference, DocEntity
from ..common.group import getDocumentGroups


class ChatResponse(TypedDict):
    status: str
    answer: str
    documents: List[str]
    followUp: List[str]


class DocInfo(TypedDict):
    objectId: str
    name: str
    score: float
    content: str


class DistillResult(TypedDict):
    answer: str
    type: str
    documents: List[DocReference]
    followUp: List[str]
    processed: int


class Distill():
    '''
    The Distill takes a series of documents and extracts answers a question
    across many documents in the required output format. This takes into
    account a variable prompt size and attempts, by counting tokens, to
    not overflow the prompt, by recursively issuing the questions across
    multiple documents.
    '''

    '''
    Privates
    '''
    _chat: ChatBase
    _maxTokens: int = 0

    def __init__(self, chat: ChatBase):
        '''
        Create all the fixed objects that we need across multiple
        invocations
        '''
        # If we do not have a chat yet, get one
        self._chat = chat

        # Get the model shape
        self._maxTokens = self._chat.modelTotalTokens

    async def adistill(
            self,
            query: str,
            docs: List[Doc],
            *,
            promptName: str,
            docFilter: DocFilter,
            returnType: str = '') -> DistillResult:
        '''
        Given a query, a set of documents retrieved by the search
        endpoint, find the requested information
        '''
        # Get the document groups
        groups = getDocumentGroups(query=query, docs=docs, docFilter=docFilter)

        # Get an empty prompt so we can size it
        prompt = Prompts.getPrompt(promptName, {
            'query': query,
            'prevResult': '',
            'returnType': returnType,
            'documents': ''
        })

        # Get the tokens in the base prompt
        baseTokens = self._chat.getTokens(prompt)

        # Walk through the document groups
        maxTokens: int = self._chat.modelTotalTokens - (baseTokens + 64)

        isComplete = False

        # Define the context
        docContext: DocInfo = {
            'objectId': '',
            'name': '',
            'score': 0.0,
            'content': ''
        }

        docBaseTokens: int = 0
        docContentTokens: int = 0

        prevAnswerContent = ''
        prevAnswerTokens: int = 0

        promptContexts: List[DocInfo] = []
        processed: int = 0
        refdocs: List[str] = []
        followUpQuestions: List[str] = []
        entities: List[DocEntity] = []

        def findGroup(groups: List[DocGroup], objectId: str) -> DocGroup | None:
            '''
            Find any chunk of a document with the give object id. This is used to
            obtain the path and permissions from. Since all chunks should be the
            same, the first one found should be good
            '''
            # Find the group in the list
            for group in groups:
                # If his is it...
                if group.objectId == objectId:
                    return group

            # Couldn't find it
            return None

        def parseResponse(content: str) -> ChatResponse:
            '''
            Clear up any missing fields and force into the ChatResponse type
            '''

            # Trim spaces
            content = content.strip()

            # If it is not json, use it directly as the result
            if content[0] != '{':
                # Build the response with it directly - the llm messed up
                response: ChatResponse = {
                    'status': 'NONE',
                    'answer': content,
                    'documents': [],
                    'followUp': []
                }
            else:
                # Parse the json
                response = cast(ChatResponse, parseJson(content))

                # Make sure it has all the fields we need
                if 'answer' not in response:
                    debug('            Distill had no answer')
                    response['answer'] = ''
                if 'documents' not in response:
                    debug('            Distill had no documents')
                    response['documents'] = []
                if 'followUp' not in response:
                    debug('            Distill had no followUp')
                    response['followUp'] = []

            # And return it
            return response

        def getTotalTokens() -> int:
            return baseTokens + prevAnswerTokens + docBaseTokens + docContentTokens

        def setContext(group: DocGroup) -> None:
            # Reset for the next prompt
            nonlocal docContext, docBaseTokens

            # Say we are distilling a document
            debug(f'        Distilling {group.parent}')

            # Reset the context
            docContext = {
                'objectId': group.objectId,
                'name': group.parent,
                'score': group.score,
                'content': ''
            }

            # Get the base tokens used by the docContext
            docBaseTokens = self._chat.getTokens(json.dumps(docContext))

        async def flushContext() -> None:
            nonlocal isComplete, refdocs
            nonlocal promptContexts, docContext
            nonlocal docBaseTokens, docContentTokens
            nonlocal prevAnswerContent, prevAnswerTokens
            nonlocal followUpQuestions, entities

            # If we have nothing to flush, then done
            if not docContentTokens or isComplete:
                return

            debug(f'            Flushing {docContext["name"]} ({getTotalTokens()}/{maxTokens})')
            debug(f'            Fixed prompt  : {docBaseTokens} tokens')
            debug(f'            Prev result   : {prevAnswerTokens} tokens')
            debug(f'            Content       : {docContentTokens} tokens')

            # Push the current content into the prompt
            promptContexts.append(docContext)

            # Get an empty prompt so we can size it
            prompt = Prompts.getPrompt(promptName, {
                'query': query,
                'prevResult': json.dumps(prevAnswerContent),
                'returnType': returnType,
                'documents': json.dumps(promptContexts)
            })

            # Call the llm
            content = await self._chat.achat(prompt)

            # Parse the output
            response = parseResponse(content)

            # Print the result we got on this pass
            debug(f'            Distill result: {response["answer"]}')

            if 'answer' in response and response['answer']:
                prevAnswerContent = response['answer']
                prevAnswerTokens = self._chat.getTokens(prevAnswerContent)

            if 'status' in response:
                match response['status']:
                    case 'DONE':
                        isComplete = True
                        pass

                    case 'CONTINUE':
                        # nothing really to do
                        pass

            # Add all the referenced documents
            if 'documents' in response:
                # Get the list of documents the the LLM referenced
                doclist = response['documents']

                # For each document
                for objectId in doclist:
                    # If it is not in the list, append it
                    if objectId not in refdocs:
                        debug(f'            Distill doc   : {objectId}')
                        refdocs.append(objectId)

            # If followup questions were returned
            if 'followUp' in response:
                # Add them until we have the max
                for followUp in response['followUp']:
                    followUpQuestions.append(followUp)

            # Reset the content tokens
            docContentTokens = 0

            # Clear the document context
            docContext['content'] = ''

            # Clear the pending contexts
            promptContexts = []

        def completeContext():
            nonlocal promptContexts

            # Say we are distilling a document
            debug(f'            Distilling for {docContext["name"]} complete')

            # Save this context, accumulate until we flush
            promptContexts.append(docContext)
            pass

        def doesFitInRemaining(tokens: int) -> bool:
            # Determines if a given chunk will fit in the remaining tokens
            if baseTokens + prevAnswerTokens + docBaseTokens + docContentTokens + tokens < maxTokens:
                return True
            else:
                return False

        def doesFitInPrompt(tokens: int) -> bool:
            # Determines if a given chunk will fit into an empty prompt
            if baseTokens + prevAnswerTokens + docBaseTokens + tokens < maxTokens:
                return True
            else:
                return False

        def addToContext(chunkId: int, tokens: int, content: str) -> None:
            nonlocal docContext, docContentTokens

            # Append the content
            if docContext is not None:
                docContext['content'] = docContext['content'] + content
            docContentTokens = docContentTokens + tokens

            # Output how much prompt space we used
            out = min(30, len(content))
            debug(f'            Adding chunk {chunkId} ({getTotalTokens()}/{maxTokens}) to context [{content[:out]}]')

        for group in groups:
            # Make the tables rise to the top
            documents = sorted(group.documents,
                               key=lambda chunk: (chunk.metadata['isTable'], chunk.score), reverse=True)

            # Reset our context out context
            setContext(group)

            # If we are complete, stop
            if isComplete:
                break

            # For each document
            for doc in documents:
                # If we are complete, stop
                if isComplete:
                    break

                # We are processing another document
                processed = processed + 1

                # Get the content
                content = safeString(doc.page_content)

                # Get the tokens in the table
                tokens = self._chat.getTokens(content)

                # Get the chunk id
                chunkId = doc.metadata['chunk']

                # If table processing or not...
                if doc.metadata['isTable']:
                    # If we can fit the entire table within a prompt, do so
                    if doesFitInRemaining(tokens):
                        addToContext(chunkId, tokens, content)
                        continue

                    # Flush the current context
                    await flushContext()

                    # If this fits in an empty prompt, which we just flushed
                    if doesFitInRemaining(tokens):
                        # Add this to the context
                        addToContext(chunkId, tokens, content)
                        continue

                    # Nope, won't fit even on an empty prompt, so we have to split the table
                    tableLines = content.split('\n')

                    # Now, get all the token count for each line
                    tableTokens: List[int] = []
                    for line in tableLines:
                        tableTokens.append(self._chat.getTokens(line + '\n'))

                    # We are going to copy 3 lines of header data
                    numberHeaderLines: int = 3

                    debug(f'            Splitting table of {tokens} tokens')

                    # Now, starting at line numberHeaderLines,
                    for index in range(numberHeaderLines, len(tableTokens)):
                        # If we are complete, stop
                        if isComplete:
                            break

                        # If it is empty
                        if not docContentTokens:
                            debug('            Adding table header')
                            # Add the first n header lines to it
                            for header in range(0, len(tableLines)):
                                # If we have copied over the max lines, done
                                if header >= numberHeaderLines:
                                    break

                                # If the line doesn't fit in the prompt, nothing we can do with
                                # it, so ignore it. This should never happen unless we could not split the table
                                if not doesFitInPrompt(tableTokens[index]):
                                    continue

                                # Add the header line to the context
                                addToContext(chunkId, tableTokens[header], tableLines[header])

                        # If this line will fit in the prompt, add it
                        if doesFitInRemaining(tableTokens[index]):
                            addToContext(chunkId, tableTokens[index], tableLines[index] + '\n')
                            continue

                        # If the line doesn't fit in the prompt, nothing we can do with
                        # it, so ignore it. This should never happen unless we could not split the table
                        if not doesFitInPrompt(tableTokens[index]):
                            continue

                        # Flush the current context
                        await flushContext()
                else:
                    # If we can fit this in the prompt, do so
                    if doesFitInRemaining(tokens):
                        addToContext(chunkId, tokens, content)
                        continue

                    # If the content doesn't fit in the prompt, nothing we can do with
                    # it, so ignore it. This should never happen unless we could we really
                    # messed up on the retrieval
                    if not doesFitInPrompt(tokens):
                        continue

                    # Doesnt fit, so flush it
                    await flushContext()

                    # Add this
                    addToContext(chunkId, tokens, content)

            # Say we are done with this document context
            completeContext()

        # Flush the final content
        await flushContext()

        # Build th result
        result: DistillResult = {
            'type': 'text',
            'answer': prevAnswerContent,
            'documents': [],
            'followUp': followUpQuestions,
            'processed': processed
        }

        # Convert objectIds into a structure
        for refdoc in refdocs:
            # Find the document group referencing it
            docGroup = findGroup(groups, refdoc)

            # If we found it
            if docGroup is not None:
                # Create a doc reference
                ref: DocReference = {
                    'objectId': refdoc,
                    'document': docGroup.parent
                }

                # And add it
                result['documents'].append(ref)

        # And return the result
        return result

    def distill(
            self,
            query: str,
            docs: List[Doc],
            *,
            promptName: str,
            docFilter: DocFilter,
            returnType: str = '') -> DistillResult:
        # Run the async version
        return asyncio.run(self.adistill(
            query,
            docs,
            promptName=promptName,
            docFilter=docFilter,
            returnType=returnType
        ))
