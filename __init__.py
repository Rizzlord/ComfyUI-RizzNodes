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
    RizzSaveImage, RizzPreviewImage, RizzLoadImage, RizzImageEffects
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
NODE_DISPLAY_NAME_MAPPINGS["RizzSaveImage"] = "💾 Save Image (Rizz)"

NODE_CLASS_MAPPINGS["RizzPreviewImage"] = RizzPreviewImage
NODE_DISPLAY_NAME_MAPPINGS["RizzPreviewImage"] = "👁️ Preview Image (Rizz)"

NODE_CLASS_MAPPINGS["RizzLoadImage"] = RizzLoadImage
NODE_DISPLAY_NAME_MAPPINGS["RizzLoadImage"] = "📂 Load Image (Rizz)"

NODE_CLASS_MAPPINGS["RizzImageEffects"] = RizzImageEffects
NODE_DISPLAY_NAME_MAPPINGS["RizzImageEffects"] = "🎨 Image Effects (Rizz)"

# Tell ComfyUI where to find our JavaScript files
WEB_DIRECTORY = "js"

try:
    import server
    from aiohttp import web
    import os

    _EASY_REQUIRED_INPUT_DEFAULTS = {
        # Use scalar defaults (not None) because easy-use's Any type compares equal
        # to core scalar types during validation, which can trigger conversions.
        "easy cleanGpuUsed": {"anything": 0},
        "easy clearCacheAll": {"anything": 0},
        "easy clearCacheKey": {"anything": 0, "cache_key": "*"},
    }

    def _patch_easy_use_missing_inputs(json_data):
        prompt = json_data.get("prompt")
        if not isinstance(prompt, dict):
            return json_data

        patched_inputs = 0
        for node_data in prompt.values():
            if not isinstance(node_data, dict):
                continue

            defaults = _EASY_REQUIRED_INPUT_DEFAULTS.get(node_data.get("class_type"))
            if not defaults:
                continue

            inputs = node_data.get("inputs")
            if not isinstance(inputs, dict):
                inputs = {}
                node_data["inputs"] = inputs

            for key, default_value in defaults.items():
                if key not in inputs:
                    inputs[key] = default_value
                    patched_inputs += 1

        if patched_inputs:
            print(f"[RizzNodes] Patched {patched_inputs} missing easy-use input(s).")

        return json_data

    if not getattr(server.PromptServer.instance, "_rizz_easy_input_patch", False):
        server.PromptServer.instance.add_on_prompt_handler(_patch_easy_use_missing_inputs)
        server.PromptServer.instance._rizz_easy_input_patch = True

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
                path = os.path.join(folder_paths.get_output_directory(), "RizzImage")
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

    @server.PromptServer.instance.routes.post("/rizz/ensure_input")
    async def ensure_input(request):
        try:
            import shutil
            import folder_paths
            data = await request.json()
            filename = data.get("filename")
            folder = data.get("folder", "None")
            custom_path = data.get("custom_path", "")
            if not filename:
                return web.json_response({"ok": False})

            # Sanitize filename
            filename = os.path.basename(filename)

            input_dir = folder_paths.get_input_directory()
            input_path = os.path.join(input_dir, filename)
            if os.path.exists(input_path):
                return web.json_response({"ok": True})

            output_root = folder_paths.get_output_directory()
            if folder == "None":
                target_folder = os.path.join(output_root, "RizzImage")
            elif folder == "Custom":
                if custom_path and os.path.isabs(custom_path):
                    target_folder = custom_path
                else:
                    target_folder = os.path.join(output_root, custom_path or "")
            else:
                target_folder = os.path.join(output_root, "RizzImage", folder)

            candidates = [
                os.path.join(target_folder, filename),
                os.path.join(output_root, "RizzImage", filename),
            ]

            src_path = next((p for p in candidates if os.path.exists(p)), None)
            if not src_path:
                return web.json_response({"ok": False})

            try:
                os.makedirs(input_dir, exist_ok=True)
                try:
                    os.symlink(src_path, input_path)
                except Exception:
                    shutil.copy2(src_path, input_path)
                return web.json_response({"ok": True})
            except Exception as e:
                print(f"[RizzNodes] ensure_input copy failed: {e}")
                return web.json_response({"ok": False})
        except Exception as e:
            print(f"[RizzNodes] ensure_input error: {e}")
            return web.json_response({"ok": False})

    @server.PromptServer.instance.routes.post("/rizz/video_first_frame")
    async def video_first_frame(request):
        try:
            import subprocess
            import folder_paths as fp
            data = await request.json()
            video_path = data.get("path", "")

            if not video_path:
                folder = data.get("folder_path", "")
                fname = data.get("file", "")
                if folder and fname:
                    if folder == "RizzVideo":
                        video_path = os.path.join(fp.get_output_directory(), "RizzVideo", fname)
                    elif folder == "RizzAudio":
                        video_path = os.path.join(fp.get_output_directory(), "RizzAudio", fname)
                    elif os.path.isabs(folder):
                        video_path = os.path.join(folder, fname)
                    else:
                        video_path = os.path.join(fp.get_output_directory(), folder, fname)

            if not video_path:
                return web.json_response({"error": "no path"}, status=400)

            if not os.path.isabs(video_path):
                for base in [fp.get_output_directory(), fp.get_input_directory(),
                             os.path.join(fp.get_output_directory(), "RizzVideo")]:
                    candidate = os.path.join(base, video_path)
                    if os.path.isfile(candidate):
                        video_path = candidate
                        break

            if not os.path.isfile(video_path):
                return web.json_response({"error": "not found"}, status=404)

            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", video_path
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            w, h, fc, fps_val = 640, 360, 100, 24.0
            if probe.returncode == 0:
                import json as _json
                info = _json.loads(probe.stdout)
                for s in info.get("streams", []):
                    if s.get("codec_type") == "video":
                        w = int(s.get("width", w))
                        h = int(s.get("height", h))
                        fc = int(s.get("nb_frames", 0)) or 100
                        r = s.get("r_frame_rate", "24/1")
                        parts = r.split("/")
                        if len(parts) == 2 and float(parts[1]) > 0:
                            fps_val = float(parts[0]) / float(parts[1])
                        break
                fmt = info.get("format", {})
                dur = float(fmt.get("duration", 0))
                if dur > 0 and fc <= 1:
                    fc = int(dur * fps_val)

            req_frame = int(data.get("frame", 0))
            seek_time = req_frame / fps_val if fps_val > 0 else 0

            frame_cmd = ["ffmpeg", "-y"]
            if seek_time > 0:
                frame_cmd.extend(["-ss", f"{seek_time:.4f}"])
            frame_cmd.extend([
                "-i", video_path,
                "-vframes", "1", "-f", "image2pipe",
                "-vcodec", "mjpeg", "-q:v", "5", "pipe:1"
            ])
            result = subprocess.run(frame_cmd, capture_output=True, timeout=10)
            if result.returncode != 0 or not result.stdout:
                return web.json_response({"error": "frame extraction failed"}, status=500)

            resp = web.Response(body=result.stdout, content_type="image/jpeg")
            resp.headers["X-Video-Width"] = str(w)
            resp.headers["X-Video-Height"] = str(h)
            resp.headers["X-Video-Frames"] = str(fc)
            resp.headers["X-Video-FPS"] = str(round(fps_val, 3))
            return resp
        except Exception as e:
            print(f"[RizzNodes] video_first_frame error: {e}")
            return web.json_response({"error": str(e)}, status=500)

except ImportError:
    print("[RizzNodes] Failed to import server for backend routes.")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("### RizzNodes: Custom nodes loaded successfully. ###")
print("### By @GhettoRizz ###")
