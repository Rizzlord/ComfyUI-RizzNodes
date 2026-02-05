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

try:
    import server
    from aiohttp import web
    import os

    @server.PromptServer.instance.routes.post("/rizz/list_files")
    async def list_files(request):
        try:
            data = await request.json()
            path = data.get("path")
            type_filter = data.get("type", "audio") # "audio" or "video"
            
            if not path:
                return web.json_response({"files": []})

            # Check for path aliases
            if path == "RizzVideo":
                import folder_paths
                path = os.path.join(folder_paths.get_output_directory(), "RizzVideo")
            elif path == "RizzAudio":
                import folder_paths
                path = os.path.join(folder_paths.get_output_directory(), "RizzAudio")
            
            if not os.path.isdir(path):
                return web.json_response({"files": []})
            
            files = []
            if type_filter == "audio":
                extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
            else: # video
                extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
            
            for f in os.listdir(path):
                if not f.startswith('.'):
                    file_path = os.path.join(path, f)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(f)
                        if ext.lower() in extensions:
                            files.append(f)
            
            return web.json_response({"files": sorted(files)})
        except Exception as e:
            print(f"[RizzNodes] Error listing files: {e}")
            return web.json_response({"files": []})

except ImportError:
    print("[RizzNodes] Failed to import server for backend routes.")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("### RizzNodes: Custom nodes loaded successfully. ###")
print("### By @GhettoRizz ###")