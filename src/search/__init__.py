# Set this. Haystack forces this to 1, or True, which loads weights
# only, but PyTorch has deprecated this functionality. We will probably
# load to much, but thats ok. Better to load too much rather than getting
# a fatal error once PyTorch removes the typed storage function
import os
from .search import Search
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = '0'
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = '0'

# Reference it
Search
