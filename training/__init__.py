# Compatibility layer for training modules
# This maintains backward compatibility for existing imports

import warnings
from llmbuilder.core.training import *

# Issue deprecation warning
warnings.warn(
    "Importing from 'training' module is deprecated. "
    "Please use 'from llmbuilder.core.training import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)