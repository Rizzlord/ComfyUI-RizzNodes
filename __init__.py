# __init__.py

# This file makes the 'RizzNodes' directory a Python package and exposes the
# necessary node mappings to ComfyUI.

# Import the node class and display name mappings from your main node file
from .rizznodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Define __all__ to specify which names are exported when `from . import *` is used.
# ComfyUI specifically looks for these two dictionaries.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# You can add a print statement for confirmation in the console when ComfyUI starts.
# This helps verify that your custom node package is being loaded.
print("### RizzNodes: Custom nodes loaded successfully. ###")
print("### By @GhettoRizz ###")