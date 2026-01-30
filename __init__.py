from .rizznodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .rizznodes_gds import RizzGDSPatcher
from .rizznodes_meshops import QuadremeshNode
from .rizznodes_textureops import RizzMakeTileable, RizzPreviewTiling
from .rizznodes_videosuit import (
    RizzLoadVideo, RizzExtractFrames, RizzVideoEffects, RizzSaveVideo,
    NODE_CLASS_MAPPINGS as VIDEO_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VIDEO_NODE_DISPLAY_NAME_MAPPINGS
)
from .rizznodes_audioops import (
    RizzAudioMixer,
    NODE_CLASS_MAPPINGS as AUDIO_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as AUDIO_NODE_DISPLAY_NAME_MAPPINGS
)

NODE_CLASS_MAPPINGS["RizzGDSPatcher"] = RizzGDSPatcher
NODE_DISPLAY_NAME_MAPPINGS["RizzGDSPatcher"] = "GDS Patcher (Rizz)"

NODE_CLASS_MAPPINGS["Quadremesh"] = QuadremeshNode
NODE_DISPLAY_NAME_MAPPINGS["Quadremesh"] = "Quadremesh (Adaptive)"

NODE_CLASS_MAPPINGS["RizzMakeTileable"] = RizzMakeTileable
NODE_DISPLAY_NAME_MAPPINGS["RizzMakeTileable"] = "Make Tileable (Rizz)"

NODE_CLASS_MAPPINGS["RizzPreviewTiling"] = RizzPreviewTiling
NODE_DISPLAY_NAME_MAPPINGS["RizzPreviewTiling"] = "Preview Tiling (Rizz)"

# Register VideoSuit nodes
NODE_CLASS_MAPPINGS.update(VIDEO_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_NODE_DISPLAY_NAME_MAPPINGS)

# Register Audio nodes
NODE_CLASS_MAPPINGS.update(AUDIO_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(AUDIO_NODE_DISPLAY_NAME_MAPPINGS)

# Tell ComfyUI where to find our JavaScript files
WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("### RizzNodes: Custom nodes loaded successfully. ###")
print("### By @GhettoRizz ###")