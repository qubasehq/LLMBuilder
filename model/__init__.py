# Compatibility layer for model modules
# This maintains backward compatibility for existing imports

import warnings
from llmbuilder.core.model import *

# Issue deprecation warning
warnings.warn(
    "Importing from 'model' module is deprecated. "
    "Please use 'from llmbuilder.core.model import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)