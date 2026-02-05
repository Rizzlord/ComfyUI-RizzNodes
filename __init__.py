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
from .rizznodes_images import (
    RizzSaveImage, RizzPreviewImage, RizzLoadImage
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

# Register Image nodes
NODE_CLASS_MAPPINGS["RizzSaveImage"] = RizzSaveImage
NODE_DISPLAY_NAME_MAPPINGS["RizzSaveImage"] = "Save Image (Rizz)"

NODE_CLASS_MAPPINGS["RizzPreviewImage"] = RizzPreviewImage
NODE_DISPLAY_NAME_MAPPINGS["RizzPreviewImage"] = "Preview Image (Rizz)"

NODE_CLASS_MAPPINGS["RizzLoadImage"] = RizzLoadImage
NODE_DISPLAY_NAME_MAPPINGS["RizzLoadImage"] = "Load Image (Rizz)"

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
            if path == "None":
                import folder_paths
                path = folder_paths.get_output_directory()
            elif path == "Custom":
                # For custom, we rely on a second "custom_path" arg if we could, 
                # but the request.json() "path" is what we get.
                # The JS will send the resolved path as "path" if it's custom.
                # Wait, if we send "Flux", logic below handles it? No, need mapping.
                pass 
            elif path in ["RizzVideo", "RizzAudio"]:
                import folder_paths
                path = os.path.join(folder_paths.get_output_directory(), path)
            elif path in ["Flux", "flux2", "qwen", "qwenedit", "sd1.5", "sdxl", "sd3", "anime"]:
                import folder_paths
                path = os.path.join(folder_paths.get_output_directory(), "RizzImage", path)
            
            if not os.path.isdir(path):
                # Try relative to output directory (standard behavior for custom paths in this node)
                import folder_paths
                out_path = os.path.join(folder_paths.get_output_directory(), path)
                if os.path.isdir(out_path):
                    path = out_path
                else:
                    return web.json_response({"files": []})
            
            files = []
            if type_filter == "audio":
                extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
            elif type_filter == "video": # video
                extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
            else: # image
                extensions = {".png", ".jpg", ".jpeg", ".webp", ".tga", ".bmp", ".tiff"}
            
            files = []
            try:
                # Use listdir for flat listing (non-recursive)
                # This prevents "None" (=output root) from showing files inside RizzImage/
                for f in os.listdir(path):
                    if os.path.isfile(os.path.join(path, f)):
                        if not f.startswith('.'):
                            _, ext = os.path.splitext(f)
                            if ext.lower() in extensions:
                                 files.append(f)
            except Exception as e:
                pass # path might not exist or be permission denied
            
            return web.json_response({"files": sorted(files)})
        except Exception as e:
            print(f"[RizzNodes] Error listing files: {e}")
            return web.json_response({"files": []})

except ImportError:
    print("[RizzNodes] Failed to import server for backend routes.")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("### RizzNodes: Custom nodes loaded successfully. ###")
print("### By @GhettoRizz ###")