from .rizznodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .rizznodes_gds import RizzGDSPatcher

NODE_CLASS_MAPPINGS["RizzGDSPatcher"] = RizzGDSPatcher
NODE_DISPLAY_NAME_MAPPINGS["RizzGDSPatcher"] = "GDS Patcher (Rizz)"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### RizzNodes: Custom nodes loaded successfully. ###")
print("### By @GhettoRizz ###")