# Compatibility layer for data modules
# This maintains backward compatibility for existing imports

import warnings
from llmbuilder.core.data import *

# Issue deprecation warning
warnings.warn(
    "Importing from 'data' module is deprecated. "
    "Please use 'from llmbuilder.core.data import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)