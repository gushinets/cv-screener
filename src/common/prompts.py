'''
Controls all the prompts - read from disk files, cached
and reloaded on demand
'''
from langchain.prompts import Prompt
from typing import Dict, Any
import glob
import os


class Prompts():
    '''
    This prompts class is a static class which controls all
    the prompts
    '''

    '''
    Privates
    '''
    _prompts: Dict[str, str] = {}
    _init: bool = False

    @staticmethod
    def loadPrompts():
        '''
        This function will read all the prompts and put them into
        the prompts dictionary
        '''
        # Clear the prompts in case we are reloading
        Prompts._prompts = {}

        # Set the root path on where to look
        rootPath = './ai/prompts'

        # Find all the prompt files
        promptFiles = glob.glob(
            './**/*', root_dir=rootPath, recursive=True)

        # Set the root key
        for promptFile in promptFiles:
            # If this is one of the cache files, skip it
            if '__pycache__' in promptFile.lower():
                continue

            # Get the full path
            promptPath = os.path.join(rootPath, promptFile)

            # If this is a directory, skip it
            if not os.path.isfile(promptPath):
                continue

            # Read the prompt file
            with open(promptPath) as f:
                promptText = f.read()
                f.close()

            # Initialize components list
            components = []

            # Iteratively split the path until it's empty
            while True:
                promptFile, component = os.path.split(promptFile)
                if component:
                    components.insert(0, component)
                else:
                    if promptFile:
                        components.insert(0, promptFile)
                    break

            # If it is less the two components, skip it
            if len(components) < 2:
                continue

            # Remove the leading .
            if components[0] == '.':
                components = components[1:]

            # Get the final component
            promptName = components.pop()

            # Make sure it ends with .txt and strip it
            if not promptName.endswith('.txt'):
                continue

            # Remove the .txt
            promptName = promptName[:len(promptFile) - 4]

            # Walk down the components
            obj = Prompts._prompts
            for component in components:
                if component not in obj:
                    obj[component] = {}
                obj = obj[component]

            # Set it
            obj[promptName] = promptText

    @staticmethod
    def getPrompt(path: str, params: Dict[str, Any] = {}):
        # If we have not loaded the prompts yet, do so now
        if not Prompts._init:
            Prompts.loadPrompts()
            Prompts._init = True

        # Get the components
        components = path.split('.')

        # Walk down the components
        obj: Any = Prompts._prompts
        for component in components:
            if component not in obj:
                raise BaseException(f'Prompt {component} not found in prompts')
            obj = obj[component]

        # Get the prompt builder
        promptBuilder = Prompt.from_template(obj)

        # Setup our prompt
        promptValue = promptBuilder.invoke(params)

        # Convert it to a string
        promptString = promptValue.to_string()

        # Return it
        return promptString
