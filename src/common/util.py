'''
Basic utility functions
'''

from typing import Any, List
from .schema import DocDict, Doc
import json


def safeString(value: str) -> str:
    '''
    Replaces all double quotes wih single quotes. This is done when
    we send a document over to the LLM as context or something so
    we don't confuse it... The prompts themselves use double quotes...
    '''

    # If it is None, return an empty string
    if value is None:
        return ''

    # Create a string from it and replace all the " with \'
    return str(value).strip().replace('"', '\'')


def parseJson(value: str) -> Any:
    '''
    Parse a string and return a json value
    '''
    # Sometimes the llm messes up and puts hard CR/LF sequences into the json strings. Don't allow that to happen
    # because our parser can't parse it - it's illegal JSON

    try:
        # Strip it
        value = value.strip()

        # Replace hard CR to nothing
        value = value.replace('\r', '')

        # Replace hard LF into nothing
        value = value.replace('\n', '')

        # Replace hard TAB into space
        value = value.replace('\t', ' ')

        # Fix it in case the llm gave us a narative
        if value.startswith('```json'):
            value = value[7:]

        if value.endswith('```'):
            value = value[0:len(value) - 3]

        v = json.loads(value)
        return v

    except Exception as e:
        debug(f'Unable to parse json ${str(e)} ${str(value)}')
        raise e


def parsePython(value: str) -> Any:
    '''
    Parse a string and return a python code snippet
    '''
    try:
        # Fix it in case the llm gave us a narative
        offset = value.find('```python')
        if offset >= 0:
            value = value[offset + 9:]
            offset = value.find('```')
            if offset >= 0:
                value = value[:offset]

        # Return it
        return value

    except Exception as e:
        debug(f'Unable to parse json {str(e)} {str(value)}')
        raise e


def docsToDicts(docDicts: List[Doc]) -> List[DocDict]:
    return []
