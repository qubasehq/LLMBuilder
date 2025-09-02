# Compatibility layer for tools modules
# This maintains backward compatibility for existing imports

import warnings
from llmbuilder.core.tools import *

# Issue deprecation warning
warnings.warn(
    "Importing from 'tools' module is deprecated. "
    "Please use 'from llmbuilder.core.tools import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)