'''
OpenAI binding for the ChatLLM
'''
import asyncio
from ...common.chat import ChatBase
from ...common.config import Config
from langchain_openai import ChatOpenAI
from tiktoken import encoding_for_model, Encoding


class Chat(ChatBase):
    '''
    Creates an OpenAI chat bot
    '''

    '''
    Privates
    '''
    _model: str = ''
    _llm: ChatOpenAI
    _tokenCounter: Encoding

    def __init__(self):
        # Get our config
        config = Config.getConfig()

        # Pull out the section name
        sectionName = config['chat']

        # Get the section
        section = config[sectionName]

        # Get the API key
        key = section['genkey']

        # Get the model to use
        self._model = section['model']

        # Init the chat base
        super().__init__()

        # Get the llm
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=key,
            temperature=0
        )

        # Get a token counter
        self._tokenCounter = encoding_for_model(self._model)

    async def achat(self, prompt: str) -> str:

        # Get the plan
        results = await self._llm.ainvoke(prompt)

        # Get the response as a string
        content = str(results.content)

        return content

    def chat(self, prompt: str) -> str:
        print("Run Chat")
        return asyncio.run(self.achat(prompt))

    def getTokens(self, value: str) -> int:
        '''
        This function will determine how many tokens, according to the model
        that the given string will take. This is used to prevent overflowing
        the prompt
        '''
        # Make sure we have a token counter
        if self._tokenCounter is None:
            raise Exception('No token counter in Chat')

        return len(self._tokenCounter.encode(value))
