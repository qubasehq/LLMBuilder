# Compatibility layer for finetune modules
# This maintains backward compatibility for existing imports

import warnings
from llmbuilder.core.finetune import *

# Issue deprecation warning
warnings.warn(
    "Importing from 'finetune' module is deprecated. "
    "Please use 'from llmbuilder.core.finetune import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)