'''
This is the global configuration file that we read once
'''
import json
import sys
import os

from typing import Dict, Any


class Config():
    '''
    The config class is a static class which loads and parses
    the aiconfig.json file
    '''

    '''
    Privates
    '''
    _config: Dict[str, Any] | None = None

    @staticmethod
    def _stripComments(json: str):
        '''
        Strips comments from a string which is in json format. Note that
        it does pay attention to quotes, but it should have the proper
        CR/LF sequences. For example:
        { "key": "value" // This is the value }
        Will not be parsed correctly as the everything on a line past the
        // will be removed, which in this case will remove the terminating
        closing brace
        '''
        # Get the lines within the json
        lines = json.split('\n')

        # walk through each line
        for lineIndex in range(0, len(lines)):
            line = lines[lineIndex]

            # If we don't have a comment, continue on
            if line.find('//') < 0:
                continue

            # Walk each chr
            quote = False
            for index in range(0, len(line) - 1):
                # Get the chr and the next chr
                chr = line[index]
                nxt = line[index + 1]

                # If this is an escape within a string, skip past it
                if chr == '\\' and quote:
                    index += 1
                    continue

                # If this is a quote, entering or leaving a quoted string
                if chr == '"':
                    quote = not quote
                    continue

                # If the chr and nxt chr are comment and we are not within a string
                if chr == '/' and nxt == '/' and not quote:
                    line = line[0: index]
                    break

            # Save the modified line
            lines[lineIndex] = line

        # Return the text
        return '\n'.join(lines)

    @staticmethod
    def getModelCacheFolder():
        '''
        If the model folder exists, we will use it as a cache. If not,
        use the default folder that hugging face useby returning None
        '''
        # Get the base directory
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Get the models folder
        folder = base + '/' + 'models'

        # If it does not exist, create it
        if not os.path.exists(folder):
            # Create the directory
            os.makedirs(folder)		

        # Return it        
        return folder

    @staticmethod
    def getConfig() -> Dict:
        '''
        Reads the config.json file and returns a dictionary
        with the values. Comments are supported and striped.
        '''
        # If it is already loaded, return it
        if Config._config is not None:
            return Config._config

        # Get the path to our filename
        file = sys.modules[__name__].__file__

        if file is None:
            raise Exception('Invalid config filename')

        # Get the path
        path = os.path.dirname(file)

        # Build the config file name
        configPath = os.path.join(path, '..', 'aiconfig.json')

        # Read the json file
        with open(configPath) as f:
            jsonStr = f.read()
            f.close()

        # Remove all the comments
        jsonClean = Config._stripComments(jsonStr)

        # parse JSON object as a dictionary
        Config._config = json.loads(jsonClean)

        # Return the config
        return Config._config
