from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, Mapping

from ...common.chat import ChatBase
from ...common.config import Config

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage
)

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import build_extra_kwargs

if TYPE_CHECKING:
    from llama_cpp import LlamaGrammar

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """
    Helper function that converts a message to the dict, which is the necessary
    input type of the llama-cpp model
    Args:
        message: The messages to pass into the model.

    Returns:
        A dictionary with the message with the necessary role as one of the entries
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: BaseMessageChunk
) -> BaseMessageChunk:
    """
    Helper function that converts an answer from an llama-cpp model into a message chunk
    Args:
        _dict: The messages received from the model
        default_class: default class of the messages received

    Returns:
        Chunk of the BaseMessage
    """
    role = _dict.get("role")
    content = _dict.get("content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role == "system" or default_class == SystemMessage:
        return SystemMessage(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """
    Helper function that converts the dict, which is the necessary
    input type of the llama-cpp model into a message.
    Args:
        _dict: The dict with the format of llama-cpp input

    Returns:
        A BaseMessage
    """
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    else:
        return ChatMessage(content=_dict["content"], role=role)


async def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
    """
    Helper function that converts an answer from an llama-cpp model into a message chunk
    Args:
        response: The model response

    Returns:
        Results of the chatting with LLM
    """
    generations = []
    for choice in response["choices"]:
        message = _convert_dict_to_message(choice["message"])
        generations.append(ChatGeneration(message=message))

    token_usage = response["usage"]
    llm_output = {"token_usage": token_usage}
    return ChatResult(generations=generations, llm_output=llm_output)


class ChatLlamaCpp(BaseChatModel):
    """llama.cpp chat completion model.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python

            import ChatLlamaCpp
            llm = ChatLlamaCpp(model_path="/path/to/llama/model")
    """

    client: Any  #: :meta private:
    repo_id: Optional[str] = None
    """The path to the Llama model repository."""

    model_path: Optional[str] = None
    """The path to the Llama model file."""

    lora_base: Optional[str] = None
    """The path to the Llama LoRA base model."""

    lora_path: Optional[str] = None
    """The path to the Llama LoRA. If None, no LoRa is loaded."""

    n_ctx: int = Field(2048, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into.
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(True, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(None, alias="n_threads")
    """Number of threads to use.
    If None, the number of threads is automatically determined."""

    n_batch: Optional[int] = Field(8, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = Field(None, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

    suffix: Optional[str] = Field(None)
    """A suffix to append to the generated text. If None, no suffix is appended."""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0.8
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling."""

    logprobs: Optional[int] = Field(None)
    """The number of logprobs to return. If None, no logprobs are returned."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    repeat_penalty: Optional[float] = 1.1
    """The penalty to apply to repeated tokens."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    last_n_tokens_size: Optional[int] = 64
    """The number of tokens to look back when applying the repeat_penalty."""

    use_mmap: Optional[bool] = True
    """Whether to keep the model loaded in RAM"""

    rope_freq_scale: float = 1.0
    """Scale factor for rope sampling."""

    rope_freq_base: float = 10000.0
    """Base frequency for rope sampling."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Any additional parameters to pass to llama_cpp.Llama."""

    streaming: bool = True
    """Whether to stream the results, token by token."""

    grammar_path: Optional[Union[str, Path]] = None
    """
    grammar_path: Path to the .gbnf file that defines formal grammars
    for constraining model outputs. For instance, the grammar can be used
    to force the model to generate valid JSON or to speak exclusively in emojis. At most
    one of grammar_path and grammar should be passed in.
    """
    grammar: Optional[Union[str, LlamaGrammar]] = None
    """
    grammar: formal grammar for constraining model outputs. For instance, the grammar
    can be used to force the model to generate valid JSON or to speak exclusively in
    emojis. At most one of grammar_path and grammar should be passed in.
    """

    verbose: bool = True
    """Print verbose output to stderr."""

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError:
            raise ImportError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )

        model_path = values["model_path"]
        repo_id = values["repo_id"]
        model_param_names = [
            "rope_freq_scale",
            "rope_freq_base",
            "lora_path",
            "lora_base",
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "use_mmap",
            "last_n_tokens_size",
            "verbose",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]

        model_params.update(values["model_kwargs"])
        if model_path is not None:
            try:
                values["client"] = Llama(model_path, **model_params)
            except Exception as e:
                raise ValueError(
                    f"Could not load Llama model from path: {model_path}. "
                    f"Received error {e}"
                )
        else:
            try:
                values["client"] = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename="*Q5_K_M.gguf",
                    **model_params)
            except Exception as e:
                raise ValueError(
                    f"Could not load Llama model from path: {model_path}. "
                    f"Received error {e}"
                )

        if values["grammar"] and values["grammar_path"]:
            grammar = values["grammar"]
            grammar_path = values["grammar_path"]
            raise ValueError(
                "Can only pass in one of grammar and grammar_path. Received "
                f"{grammar=} and {grammar_path=}."
            )
        elif isinstance(values["grammar"], str):
            values["grammar"] = LlamaGrammar.from_string(values["grammar"])
        elif values["grammar_path"]:
            values["grammar"] = LlamaGrammar.from_file(values["grammar_path"])
        else:
            pass
        return values

    @root_validator(pre=True)
    def build_model_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling llama_cpp."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_sequences": self.stop,  # key here is convention among LLM classes
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
        }  # type: Dict[str, Any]
        if self.grammar:
            params["grammar"] = self.grammar
        return params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_path": self.model_path}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chatllamacpp"

    def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters in format needed by llama_cpp.

        Args:
            stop (Optional[List[str]]): List of stop sequences for llama_cpp.

        Returns:
            Dictionary containing the combined parameters.
        """

        # Raise error if stop sequences are in both input and default params
        if self.stop and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")

        params = self._default_params

        # llama_cpp expects the "stop" key not this, so we remove it:
        params.pop("stop_sequences")

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stop or stop or []

        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call the Llama model and return the output.

        Args:
            messages: The list of messages to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.
        """
        if self.streaming:
            # If streaming is enabled, we use the stream
            # method that yields as they are generated
            # and return the combined strings from the first choices's text:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        else:
            params = self._get_parameters(stop)
            params = {**params, **kwargs}
            result = self.client.create_chat_completion(messages=messages, stream=True, **params)
            return _create_chat_result(result)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            messages: The list of messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens in messages being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See llama-cpp-python docs and below for more.

        Example:
            .. code-block:: python

                import ChatLlamaCpp
                llm = ChatLlamaCpp(
                    model_path="/path/to/local/model.bin",
                    temperature = 0.5
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                        stop=["'","\n"]):
                    result = chunk["choices"][0]
                    print(result["text"], end='', flush=True)

        """
        params = {**self._get_parameters(stop), **kwargs}
        result = self.client.create_chat_completion(
            messages=[_convert_message_to_dict(m) for m in messages], stream=True, **params)
        default_chunk_class = AIMessageChunk
        for part in result:
            for choice in part["choices"]:
                chunk = _convert_delta_to_message_chunk(
                    choice["delta"], default_chunk_class
                )
                default_chunk_class = chunk.__class__
                yield ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content)

    def get_num_tokens(self, text: str) -> int:
        tokenized_text = self.client.tokenize(text.encode("utf-8"))
        return len(tokenized_text)


class Chat(ChatBase):
    '''
    Creates an OpenAI chat bot
    '''

    def __init__(self):
        # Get our config
        config = Config.getConfig()

        # Pull out the section name
        sectionName = config['chat']

        # Get the section
        section = config[sectionName]

        # Get the model to use
        self._model = section['model']

        # Init the chat base
        super().__init__()

        # Get the llm
        if 'modelPath' in section.keys():
            self._llm = ChatLlamaCpp(
                model_path=section['modelPath'],
                n_gpu_layers=-1,
                temperature=0,
                chat_format=section['chatFormat'],
                n_ctx=section['modelTotalTokens']
            )
        else:
            self._llm = ChatLlamaCpp(
                repo_id=section['model'],
                n_gpu_layers=-1,
                temperature=0,
                chat_format=section['chatFormat'],
                n_ctx=section['modelTotalTokens']
            )

    def getTokens(self, value: str) -> int:
        '''
        This function will determine how many tokens, according to the model
        that the given string will take. This is used to prevent overflowing
        the prompt
        '''

        return self._llm.get_num_tokens(value)

    async def achat(self, prompt: str) -> str:
        # debug(f'Prompt: {prompt}')

        # Get the plan
        results = await self._llm.ainvoke(prompt)

        # Get the response as a string
        content = str(results.content)

        # debug(f'Results: {content}')
        return content

    def chat(self, prompt: str) -> str:
        # debug(f'Prompt: {prompt}')

        # Get the plan
        results = self._llm.invoke(prompt)

        # Get the response as a string
        content = str(results.content)

        # debug(f'Results: {content}')
        return content
