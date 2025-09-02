# Compatibility layer for eval modules
# This maintains backward compatibility for existing imports

import warnings
from llmbuilder.core.eval import *

# Issue deprecation warning
warnings.warn(
    "Importing from 'eval' module is deprecated. "
    "Please use 'from llmbuilder.core.eval import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)