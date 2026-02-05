"""
RizzNodes VideoSuit - Video processing nodes for ComfyUI
Provides: Load Video, Extract Frames, Video Effects, Save Video
"""

import os
import sys
import subprocess
import tempfile
import uuid
import json
import numpy as np
import torch
import folder_paths
from PIL import Image

# Try importing av for video handling
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("[RizzNodes VideoSuit] Warning: PyAV not installed. Some features may be limited.")


# ============================================================================
# VIDEO type definition
# ============================================================================
# VIDEO = dict with:
#   path: str - full path to video file
#   fps: float - frames per second
#   duration: float - duration in seconds
#   width: int - video width
#   height: int - video height
#   frame_count: int - total frames
#   has_audio: bool - whether video has audio track


def get_video_info(video_path):
    """Extract video metadata using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        
        video_stream = None
        has_audio = False
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and video_stream is None:
                video_stream = stream
            if stream.get('codec_type') == 'audio':
                has_audio = True
        
        if video_stream is None:
            raise ValueError("No video stream found")
        
        # Parse FPS (can be "24/1" or "29.97" format)
        fps_str = video_stream.get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 24.0
        else:
            fps = float(fps_str)
        
        duration = float(data.get('format', {}).get('duration', 0))
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        frame_count = int(video_stream.get('nb_frames', 0))
        
        # If frame_count not available, estimate from duration
        if frame_count == 0 and duration > 0:
            frame_count = int(duration * fps)
        
        return {
            'path': video_path,
            'fps': fps,
            'duration': duration,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'has_audio': has_audio
        }
    except Exception as e:
        print(f"[RizzNodes VideoSuit] Error getting video info: {e}")
        return {
            'path': video_path,
            'fps': 24.0,
            'duration': 0.0,
            'width': 0,
            'height': 0,
            'frame_count': 0,
            'has_audio': False
        }


def extract_frame_at_time(video_path, time_seconds):
    """Extract a single frame at specified time using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-ss', str(time_seconds), '-i', video_path,
            '-vframes', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode == 0 and len(result.stdout) > 0:
            from io import BytesIO
            img = Image.open(BytesIO(result.stdout))
            img = img.convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)
    except Exception as e:
        print(f"[RizzNodes VideoSuit] Error extracting frame: {e}")
    
    # Return a blank frame on error
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


# ============================================================================
# Blend Mode Functions (Photoshop-style)
# ============================================================================

def blend_normal(base, overlay, opacity):
    return base * (1 - opacity) + overlay * opacity

def blend_multiply(base, overlay, opacity):
    result = base * overlay
    return base * (1 - opacity) + result * opacity

def blend_screen(base, overlay, opacity):
    result = 1 - (1 - base) * (1 - overlay)
    return base * (1 - opacity) + result * opacity

def blend_overlay(base, overlay, opacity):
    mask = base < 0.5
    result = np.where(mask, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))
    return base * (1 - opacity) + result * opacity

def blend_soft_light(base, overlay, opacity):
    result = np.where(
        overlay < 0.5,
        base - (1 - 2 * overlay) * base * (1 - base),
        base + (2 * overlay - 1) * (np.sqrt(base) - base)
    )
    return base * (1 - opacity) + result * opacity

def blend_hard_light(base, overlay, opacity):
    mask = overlay < 0.5
    result = np.where(mask, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))
    return base * (1 - opacity) + result * opacity

def blend_color_dodge(base, overlay, opacity):
    result = np.clip(base / (1 - overlay + 1e-7), 0, 1)
    return base * (1 - opacity) + result * opacity

def blend_color_burn(base, overlay, opacity):
    result = np.clip(1 - (1 - base) / (overlay + 1e-7), 0, 1)
    return base * (1 - opacity) + result * opacity

def blend_darken(base, overlay, opacity):
    result = np.minimum(base, overlay)
    return base * (1 - opacity) + result * opacity

def blend_lighten(base, overlay, opacity):
    result = np.maximum(base, overlay)
    return base * (1 - opacity) + result * opacity

def blend_difference(base, overlay, opacity):
    result = np.abs(base - overlay)
    return base * (1 - opacity) + result * opacity

def blend_exclusion(base, overlay, opacity):
    result = base + overlay - 2 * base * overlay
    return base * (1 - opacity) + result * opacity

BLEND_MODES = {
    'Normal': blend_normal,
    'Multiply': blend_multiply,
    'Screen': blend_screen,
    'Overlay': blend_overlay,
    'Soft Light': blend_soft_light,
    'Hard Light': blend_hard_light,
    'Color Dodge': blend_color_dodge,
    'Color Burn': blend_color_burn,
    'Darken': blend_darken,
    'Lighten': blend_lighten,
}


# ============================================================================
# Node 1: RizzLoadVideo
# ============================================================================

class RizzLoadVideo:
    """Load a video file by scanning a folder and selecting from a list."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "RizzVideo", "multiline": False}),
                "file": (["None"], {"default": "None"}),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "filename", "full_path")
    FUNCTION = "load_video"
    CATEGORY = "RizzNodes/Video"
    
    @classmethod
    def IS_CHANGED(cls, folder_path="", file=""):
        if folder_path and file and file != "None":
            path = os.path.join(folder_path, file)
            if os.path.exists(path):
                return os.path.getmtime(path)
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
    
    def load_video(self, folder_path, file):
        if not file or file == "None" or not folder_path:
            raise ValueError("Please select a video file")
        
        if not file or file == "None" or not folder_path:
            raise ValueError("Please select a video file")
        
        # Helper to resolve "RizzVideo" to output/RizzVideo
        if folder_path == "RizzVideo":
            video_path = os.path.join(folder_paths.get_output_directory(), "RizzVideo", file)
        else:
            video_path = os.path.join(folder_path, file)
            
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_info = get_video_info(video_path)
        filename = os.path.basename(video_path)
        
        return (video_info, filename, video_path)


# ============================================================================
# Node 1.5: RizzLoadAudio
# ============================================================================

class RizzLoadAudio:
    """Load an audio file by scanning a folder and selecting from a list."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "RizzAudio", "multiline": False}),
                "file": (["None"], {"default": "None"}), 
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration (seconds)")
    FUNCTION = "load_audio"
    CATEGORY = "RizzNodes/Audio"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, folder_path="", file=""):
        if folder_path and file and file != "None":
            path = os.path.join(folder_path, file)
            if os.path.exists(path):
                return os.path.getmtime(path)
        return float("nan")

    def load_audio(self, folder_path, file):
        if not file or file == "None" or not folder_path:
            return (None, 0.0)
        
        if not file or file == "None" or not folder_path:
            return (None, 0.0)
        
        # Helper to resolve "RizzAudio" to output/RizzAudio
        if folder_path == "RizzAudio":
            path = os.path.join(folder_paths.get_output_directory(), "RizzAudio", file)
        else:
            path = os.path.join(folder_path, file)
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
            
        try:
            import soundfile as sf
            # Soundfile reads as [samples, channels] or [samples]
            waveform, sample_rate = sf.read(path)
            
            # Calculate duration in seconds
            duration = waveform.shape[0] / sample_rate
            
            # Convert to tensor
            waveform = torch.from_numpy(waveform).float()
            
            # Normalize to [batch, channels, samples]
            if waveform.dim() == 1:
                # [samples] -> [1, 1, samples]
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.dim() == 2:
                # [samples, channels] -> [channels, samples]
                waveform = waveform.t().unsqueeze(0)
                
            # Standard ComfyUI AUDIO format: {"waveform": tensor [batch, channels, samples], "sample_rate": int}
            return ({"waveform": waveform, "sample_rate": sample_rate}, duration)
        except Exception as e:
            print(f"[RizzNodes] Error loading audio from {path}: {e}")
            raise ValueError(f"Failed to load audio: {e}")


# ============================================================================
# Node 2: RizzExtractFrames
# ============================================================================

class RizzExtractFrames:
    """Extract first and last frames from a video."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "IMAGE", "IMAGE")
    RETURN_NAMES = ("video", "first_frame", "last_frame")
    FUNCTION = "extract_frames"
    CATEGORY = "RizzNodes/Video"
    
    def extract_frames(self, video):
        video_path = video['path']
        duration = video['duration']
        frame_count = video.get('frame_count', 0)
        fps = video.get('fps', 24.0)
        
        # Extract first frame (at 0 seconds)
        first_frame = extract_frame_at_time(video_path, 0.0)
        
        # Extract last frame
        # Better strategy: calculate timestamp of the last frame using frame count
        if frame_count > 1 and fps > 0:
            # Time of the start of the last frame
            last_time = (frame_count - 1) / fps
        else:
            # Fallback: seek to a bit before end
            # Using 0.5s buffer is safer than 0.1s to ensure we hit a frame
            last_time = max(0, duration - 0.1)
            
        last_frame = extract_frame_at_time(video_path, last_time)
        
        # If last frame failed (black frame check), try seeking back a bit more
        is_black = torch.all(last_frame == 0)
        if is_black and duration > 1.0:
            print("[RizzNodes] Last frame extraction failed, retrying earlier...")
            # Try 1 second before end
            retry_time = max(0, duration - 1.0)
            last_frame = extract_frame_at_time(video_path, retry_time)
        
        return (video, first_frame, last_frame)


# ============================================================================
# Node 3: RizzVideoEffects
# ============================================================================

MAX_AUDIO_SLOTS = 10
MAX_IMAGE_SLOTS = 5

class RizzVideoEffects:
    """Apply effects to video: audio mixing, image overlay, speed, color adjustments.
    Supports dynamic audio/image counts for flexible multi-layer compositing."""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "video": ("VIDEO",),
                "audio_count": ("INT", {
                    "default": 0, "min": 0, "max": MAX_AUDIO_SLOTS, "step": 1,
                    "tooltip": "Number of audio tracks to mix (0-10). Each gets its own start time and volume."
                }),
                "image_count": ("INT", {
                    "default": 0, "min": 0, "max": MAX_IMAGE_SLOTS, "step": 1,
                    "tooltip": "Number of image overlays (0-5). Each gets its own blend mode, opacity, and position."
                }),
            },
            "optional": {
                # Speed/reverse
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Playback speed. 0.5 = slow motion, 2.0 = double speed."}),
                "interpolation_mode": (["None", "Frame Blend", "Optical Flow"], {
                    "default": "None",
                    "tooltip": "Interpolation method for slow motion (speed < 1.0). 'Frame Blend' is faster, 'Optical Flow' is very slow but smoother."
                }),
                "reverse": ("BOOLEAN", {"default": False, "tooltip": "Reverse video playback."}),
                # Fade
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Fade in duration in seconds (from black)."}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Fade out duration in seconds (to black)."}),
                # Color adjustments
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Brightness adjustment. -1 = black, 0 = normal, 1 = white."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Contrast adjustment. 1.0 = normal."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Saturation adjustment. 0 = grayscale, 1 = normal."}),
                # Audio duration
                "end_with_audio": ("BOOLEAN", {"default": False,
                    "tooltip": "If True, extend video to match longest audio. If False, cut audio when video ends."}),
            }
        }
        
        # Generate dynamic audio inputs (up to MAX_AUDIO_SLOTS)
        for i in range(1, MAX_AUDIO_SLOTS + 1):
            inputs["optional"][f"audio_{i}"] = ("AUDIO",)
            inputs["optional"][f"audio_{i}_start"] = ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                "tooltip": f"Audio {i} start time in seconds."
            })
            inputs["optional"][f"audio_{i}_volume"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                "tooltip": f"Audio {i} volume. 1.0 = normal, 2.0 = double."
            })
        
        # Generate dynamic image overlay inputs (up to MAX_IMAGE_SLOTS)
        blend_modes = list(BLEND_MODES.keys())
        position_modes = ["Stretched", "Tiled", "Center", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        
        for i in range(1, MAX_IMAGE_SLOTS + 1):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
            inputs["optional"][f"image_{i}_blend"] = (blend_modes, {
                "default": "Normal",
                "tooltip": f"Blend mode for image {i}."
            })
            inputs["optional"][f"image_{i}_opacity"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": f"Opacity for image {i}. 0 = transparent, 1 = fully visible."
            })
            inputs["optional"][f"image_{i}_position"] = (position_modes, {
                "default": "Stretched",
                "tooltip": f"Position/sizing mode for image {i}."
            })
            inputs["optional"][f"image_{i}_tile_scale"] = ("FLOAT", {
                "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                "tooltip": f"Scale factor for tiled image {i}. Smaller = more tiles."
            })
        
        return inputs
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "apply_effects"
    CATEGORY = "RizzNodes/Video"
    
    def apply_effects(self, video, audio_count=1, image_count=0,
                      speed=1.0, reverse=False, fade_in=0.0, fade_out=0.0,
                      brightness=0.0, contrast=1.0, saturation=1.0,
                      end_with_audio=True, interpolation_mode="None", **kwargs):
        """Apply video effects with dynamic audio/image inputs from kwargs."""
        
        input_path = video['path']
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"rizz_effects_{uuid.uuid4().hex}.mp4")
        
        # Build ffmpeg filter chain
        filters = []
        audio_filters = []
        
        # Speed adjustment
        if speed != 1.0:
            # Interpolation logic (only useful when slowing down, i.e., speed < 1.0)
            if interpolation_mode != "None":
                # Calculate target FPS to maintain smoothness at the new duration
                # If speed is 0.5 (2x slow mo), we want double the frames
                input_fps = video.get('fps', 24.0)
                target_fps = input_fps / speed
                
                if interpolation_mode == "Frame Blend":
                    filters.append(f"minterpolate='mi_mode=blend:fps={target_fps}'")
                elif interpolation_mode == "Optical Flow":
                    # mci: motion compensated interpolation
                    # mc_mode=aobmc: adaptive overlapping block motion compensation (higher quality)
                    filters.append(f"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={target_fps}'")
            
            filters.append(f"setpts={1/speed}*PTS")
            
            # atempo only supports 0.5 to 2.0, so we may need multiple stages
            curr_speed = speed
            while curr_speed < 0.5:
                audio_filters.append("atempo=0.5")
                curr_speed /= 0.5
            while curr_speed > 2.0:
                audio_filters.append("atempo=2.0")
                curr_speed /= 2.0
            audio_filters.append(f"atempo={curr_speed}")
        
        # Reverse
        if reverse:
            filters.append("reverse")
            audio_filters.append("areverse")
        
        # Color adjustments using eq filter
        eq_parts = []
        if brightness != 0.0:
            eq_parts.append(f"brightness={brightness}")
        if contrast != 1.0:
            eq_parts.append(f"contrast={contrast}")
        if saturation != 1.0:
            eq_parts.append(f"saturation={saturation}")
        if eq_parts:
            filters.append(f"eq={':'.join(eq_parts)}")
        
        # Fade effects (collect these to apply at the end of the chain)
        audio_fade_filters = []
        if fade_in > 0:
            filters.append(f"fade=t=in:st=0:d={fade_in}")
            audio_fade_filters.append(f"afade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            duration = video['duration'] / speed if speed != 1.0 else video['duration']
            fade_start = max(0, duration - fade_out - 0.05)
            filters.append(f"fade=t=out:st={fade_start}:d={fade_out}")
            audio_fade_filters.append(f"afade=t=out:st={fade_start}:d={fade_out}")
        
        # Collect overlay images (processed in order based on image_count)
        overlay_images = []
        for i in range(1, min(image_count, MAX_IMAGE_SLOTS) + 1):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                overlay_images.append({
                    'image': img,
                    'blend': kwargs.get(f"image_{i}_blend", "Normal"),
                    'opacity': kwargs.get(f"image_{i}_opacity", 1.0),
                    'position': kwargs.get(f"image_{i}_position", "Stretched"),
                    'tile_scale': kwargs.get(f"image_{i}_tile_scale", 1.0),
                    'index': i
                })
        
        # Build complex filter for overlays
        complex_filter = []
        width, height = video['width'], video['height']
        overlay_files = []
        
        # Process overlay images
        if overlay_images:
            # Map blend modes to ffmpeg
            ffmpeg_blend_map = {
                'Normal': 'normal', 'Multiply': 'multiply', 'Screen': 'screen',
                'Overlay': 'overlay', 'Soft Light': 'softlight', 'Hard Light': 'hardlight',
                'Color Dodge': 'dodge', 'Color Burn': 'burn', 'Darken': 'darken',
                'Lighten': 'lighten',
            }
            position_map = {
                "Center": "(W-w)/2:(H-h)/2", "Top-Left": "0:0", "Top-Right": "W-w:0",
                "Bottom-Left": "0:H-h", "Bottom-Right": "W-w:H-h", "Stretched": "0:0", "Tiled": "0:0"
            }
            
            # Save all overlay images and store dimensions
            for ov in overlay_images:
                overlay_path = os.path.join(temp_dir, f"overlay_{ov['index']}_{uuid.uuid4().hex}.png")
                img_data = ov['image']
                if img_data.dim() == 4:
                    img_np = img_data[0].cpu().numpy()
                else:
                    img_np = img_data.cpu().numpy()
                img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                
                # Store dimensions for tiling calculation
                ov['img_width'] = img_np.shape[1]  # width
                ov['img_height'] = img_np.shape[0]  # height
                
                if img_np.shape[-1] == 4:
                    Image.fromarray(img_np, 'RGBA').save(overlay_path)
                else:
                    Image.fromarray(img_np, 'RGB').save(overlay_path)
                overlay_files.append(overlay_path)
            
            # Build overlay chain - each overlay composited on top of previous
            if filters:
                complex_filter.append(f"[0:v]{','.join(filters)},format=rgba[v0]")
            else:
                complex_filter.append(f"[0:v]format=rgba[v0]")
            
            current_label = "v0"
            for idx, ov in enumerate(overlay_images):
                input_idx = idx + 1  # overlay images start at input 1
                ffmpeg_blend = ffmpeg_blend_map.get(ov['blend'], 'normal')
                opacity = ov['opacity']
                
                # Scale overlay
                tile_scale = ov.get('tile_scale', 1.0)
                position = ov['position']
                
                if position == "Stretched":
                    scale_filter = f"[{input_idx}:v]scale={width}:{height},format=rgba"
                    overlay_pos = "x=0:y=0"
                elif position == "Tiled":
                    # Calculate tile dimensions and counts in Python to avoid ffmpeg expression errors
                    import math
                    
                    img_w = ov['img_width']
                    img_h = ov['img_height']
                    
                    # Target dimension for each tile
                    target_w = int(img_w * tile_scale)
                    target_h = int(img_h * tile_scale)
                    
                    # Ensure minimum size to avoid div by zero
                    target_w = max(1, target_w)
                    target_h = max(1, target_h)
                    
                    # Calculate number of tiles needed to cover the video
                    tiles_x = math.ceil(width / target_w)
                    tiles_y = math.ceil(height / target_h)
                    total_tiles = tiles_x * tiles_y
                    
                    # tile filter expects N frames to create a grid. We have 1 frame (the image).
                    # We need to loop the single frame total_tiles-1 times to feed the tile filter.
                    scale_filter = (
                        f"[{input_idx}:v]"
                        f"loop=loop={total_tiles-1}:size=1:start=0,"
                        f"scale={target_w}:{target_h},"
                        f"tile={tiles_x}x{tiles_y},"
                        f"crop={width}:{height}:0:0,"
                        f"format=rgba"
                    )
                    overlay_pos = "x=0:y=0"
                else:
                    # For positioning modes, keep original size (capped to video size)
                    scale_filter = f"[{input_idx}:v]scale='min({width},iw)':'min({height},ih)',format=rgba"
                    
                    # Set position based on mode
                    if position == "Center":
                        overlay_pos = "x=(W-w)/2:y=(H-h)/2"
                    elif position == "Top-Left":
                        overlay_pos = "x=0:y=0"
                    elif position == "Top-Right":
                        overlay_pos = "x=W-w:y=0"
                    elif position == "Bottom-Left":
                        overlay_pos = "x=0:y=H-h"
                    elif position == "Bottom-Right":
                        overlay_pos = "x=W-w:y=H-h"
                    else:
                        overlay_pos = "x=0:y=0"
                
                ovr_label = f"ovr{idx}"
                out_label = f"v{idx + 1}"
                
                # Apply opacity
                complex_filter.append(f"{scale_filter},colorchannelmixer=aa={opacity}[{ovr_label}]")
                
                # Overlay
                if ffmpeg_blend == 'normal':
                    complex_filter.append(f"[{current_label}][{ovr_label}]overlay={overlay_pos}:format=auto[{out_label}]")
                else:
                    # Blend mode requires same-size inputs, pad if needed
                    # When using blend modes, we first create a blended version of the whole frame
                    # and then overlay it back onto the original frame using the image's alpha.
                    # This prevents transparency issues that turn green in YUV conversion.
                    
                    pad_x = "0"
                    pad_y = "0"
                    if "x=" in overlay_pos:
                        parts = overlay_pos.split(":")
                        for p in parts:
                            if p.startswith("x="):
                                pad_x = p[2:]
                            if p.startswith("y="):
                                pad_y = p[2:]
                    
                    pad_x = pad_x.replace("W", "ow").replace("H", "oh").replace("w", "iw").replace("h", "ih")
                    pad_y = pad_y.replace("W", "ow").replace("H", "oh").replace("w", "iw").replace("h", "ih")
                    
                    # 1. Create a padded version of the image with its alpha
                    complex_filter[-1] = f"{scale_filter},colorchannelmixer=aa={opacity},pad={width}:{height}:{pad_x}:{pad_y}:color=black@0[{ovr_label}]"
                    
                    # 2. Blend the video with the padded image
                    # The result of blend often has transparency where the padding was.
                    blend_label = f"bld{idx}"
                    complex_filter.append(f"[{current_label}][{ovr_label}]blend=all_mode={ffmpeg_blend}:all_opacity={opacity}[{blend_label}]")
                    
                    # 3. Overlay the blended result onto the video frames
                    # This restores the background and produces an opaque result.
                    complex_filter.append(f"[{current_label}][{blend_label}]overlay=format=auto[{out_label}]")
                
                current_label = out_label
            
            # Final format conversion
            complex_filter.append(f"[{current_label}]format=yuv420p[vout]")
        else:
            # No overlays
            if filters:
                complex_filter.append(f"[0:v]{','.join(filters)}[vout]")
        
        # Build ffmpeg command
        cmd = ['ffmpeg', '-y', '-i', input_path]
        
        # Add overlay image inputs
        for overlay_path in overlay_files:
            cmd.extend(['-i', overlay_path])
        
        # Collect audio inputs (processed based on audio_count)
        audio_data = []
        for i in range(1, min(audio_count, MAX_AUDIO_SLOTS) + 1):
            audio = kwargs.get(f"audio_{i}")
            if audio is not None:
                audio_data.append({
                    'audio': audio,
                    'start': kwargs.get(f"audio_{i}_start", 0.0),
                    'volume': kwargs.get(f"audio_{i}_volume", 1.0),
                    'index': i
                })
        
        # Handle audio inputs
        audio_inputs = []
        audio_mix_parts = []
        audio_files = []
        
        # Original video audio (if it has audio)
        if video['has_audio']:
            audio_mix_parts.append(f"[0:a]volume=1.0[a0]")
            audio_inputs.append("[a0]")
        
        # Add each audio track
        for ad in audio_data:
            audio_file = os.path.join(temp_dir, f"audio{ad['index']}_{uuid.uuid4().hex}.wav")
            self._save_audio_to_file(ad['audio'], audio_file)
            audio_files.append(audio_file)
            cmd.extend(['-i', audio_file])
            audio_idx = len(overlay_files) + len(audio_files)  # Account for video + overlays
            delay_ms = int(ad['start'] * 1000)
            label = f"a{len(audio_inputs)}"
            audio_mix_parts.append(f"[{audio_idx}:a]adelay={delay_ms}|{delay_ms},volume={ad['volume']}[{label}]")
            audio_inputs.append(f"[{label}]")
        
        # Build final filter complex
        filter_complex_str = ""
        
        if complex_filter:
            filter_complex_str += ";".join(complex_filter)
        
        if audio_mix_parts:
            # First, process original audio with speed/reverse filters if present
            if video['has_audio'] and audio_filters:
                # We need to inject this into the filter complex BEFORE mixing
                # audio_mix_parts[0] should be "[0:a]volume=1.0[a0]" if video['has_audio'] is true
                # We'll re-build audio_mix_parts and audio_inputs to be cleaner
                
                new_audio_mix_parts = []
                new_audio_inputs = []
                
                # Original video audio with filters applied
                new_audio_mix_parts.append(f"[0:a]{','.join(audio_filters)},volume=1.0[a0]")
                new_audio_inputs.append("[a0]")
                
                # Add the rest of the audio tracks
                for i in range(1, len(audio_inputs)):
                    part = audio_mix_parts[i]
                    # Part looks like "[audio_idx:a]adelay=...,volume=...[label]"
                    # We need the [label] part for the mix
                    label_start = part.rfind('[')
                    if label_start != -1:
                        new_audio_mix_parts.append(part)
                        new_audio_inputs.append(part[label_start:])
                
                audio_mix_parts = new_audio_mix_parts
                audio_inputs = new_audio_inputs

            if filter_complex_str:
                filter_complex_str += ";"
            filter_complex_str += ";".join(audio_mix_parts)
            
            # Determine audio duration mode
            duration_mode = "longest" if end_with_audio else "shortest"
            
            if len(audio_inputs) > 1:
                # Mix multiple audio streams
                mix_out = "amixout"
                filter_complex_str += f";{''.join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration={duration_mode}[{mix_out}]"
            else:
                # Single audio stream (might be filtered or just volume adjusted)
                mix_out = audio_inputs[0].replace('[', '').replace(']', '')
            
            # Apply fade filters to the mixed output
            if audio_fade_filters:
                filter_complex_str += f";[{mix_out}]{','.join(audio_fade_filters)}[aout]"
            else:
                # If no fades, just rename to aout
                # If we didn't mix, mix_out is already the label we want to map
                if mix_out != "aout":
                    filter_complex_str += f";[{mix_out}]anull[aout]"
        elif video['has_audio'] and (audio_filters or audio_fade_filters):
            # No mixed audio, but original audio needs filters
            all_a_filters = audio_filters + audio_fade_filters
            if filter_complex_str: filter_complex_str += ";"
            filter_complex_str += f"[0:a]{','.join(all_a_filters)}[aout]"
            audio_mix_parts = ["processed_audio"] # Flag to ensure -map [aout] is used
        
        # Add filter complex to command
        if filter_complex_str:
            cmd.extend(['-filter_complex', filter_complex_str])
            
            if complex_filter:
                cmd.extend(['-map', '[vout]'])
            else:
                cmd.extend(['-map', '0:v'])
            
            if audio_mix_parts:
                cmd.extend(['-map', '[aout]'])
            elif video['has_audio']:
                cmd.extend(['-map', '0:a'])
        elif filters:
            cmd.extend(['-vf', ','.join(filters)])
            
        # Duration handling
        # If end_with_audio is False, we CUT the video/audio at the exact video duration
        # If end_with_audio is True, we let ffmpeg run (it will extend to longest stream if we used 'longest' in amix)
        # Note: amix 'longest' keeps audio going. Video stream behavior depends on inputs.
        # But if we want to ensure video cut-off, we must use -t.
        
        if not end_with_audio:
            # Force output duration to match processed video duration (original video / speed)
            video_dur = video['duration']
            if speed != 1.0 and speed > 0:
                video_dur = video_dur / speed
            cmd.extend(['-t', str(video_dur)])
        
        # Output settings
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ])
        
        # Run ffmpeg
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[RizzNodes VideoSuit] FFmpeg error: {result.stderr}")
                # If complex filter failed, try simpler approach
                simple_cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                    output_path
                ]
                subprocess.run(simple_cmd, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            print("[RizzNodes VideoSuit] FFmpeg timeout - video processing took too long")
            return (video,)
        
        # Return new video info
        if os.path.exists(output_path):
            return (get_video_info(output_path),)
        else:
            return (video,)
    
    def _save_audio_to_file(self, audio_data, filepath):
        """Save ComfyUI AUDIO type to a WAV file."""
        import soundfile as sf
        
        if isinstance(audio_data, dict):
            waveform = audio_data['waveform']
            sample_rate = audio_data['sample_rate']
        else:
            # Fallback tuple format
            sample_rate, waveform = audio_data
        
        # Handle tensor shapes
        if hasattr(waveform, 'cpu'):
            waveform = waveform.cpu()
        if hasattr(waveform, 'numpy'):
            waveform = waveform.numpy()
        
        # Normalize shape to [time, channels]
        if waveform.ndim == 3:
            waveform = waveform[0]  # Remove batch dim
        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T  # [channels, time] -> [time, channels]
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)
        
        sf.write(filepath, waveform, sample_rate)


# ============================================================================
# Node 4: RizzSaveVideo
# ============================================================================

class RizzSaveVideo:
    """Save video with customizable encoding settings."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "RizzVideo/Vid", "tooltip": "Prefix for the output filename. Counter will be appended automatically."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Frames per second for the output video. Ignored if use_source_fps is True."}),
            },
            "optional": {
                "codec": (["libx264", "libx265", "mpeg4", "libvpx-vp9"], {"default": "libx264", "tooltip": "Video codec. libx264 (H.264) is most compatible. libx265 (H.265/HEVC) has better compression. libvpx-vp9 for WebM format."}),
                "quality_crf": ("INT", {"default": 23, "min": 0, "max": 51, "tooltip": "Constant Rate Factor: 0 = lossless, 23 = default, 51 = worst quality. Lower = better quality, larger file."}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "medium", "tooltip": "Encoding speed preset. Slower = better compression but longer encoding time."}),
                "pixel_format": (["yuv420p", "yuv444p"], {"default": "yuv420p", "tooltip": "yuv420p is widely compatible. yuv444p preserves more color detail but less compatible."}),
                "audio_codec": (["aac", "mp3"], {"default": "aac", "tooltip": "Audio codec. AAC is most compatible with MP4."}),
                "audio_bitrate": (["64k", "96k", "128k", "192k", "256k", "320k"], {"default": "192k", "tooltip": "Audio bitrate. Higher = better quality. 192k is good quality, 320k is near-lossless."}),
                "use_source_fps": ("BOOLEAN", {"default": True, "tooltip": "If True, uses the original video's FPS. If False, uses the fps setting above."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("filepath", "video")
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "RizzNodes/Video"
    
    def save_video(self, video, filename_prefix, fps, codec="libx264", quality_crf=23,
                   preset="medium", pixel_format="yuv420p", audio_codec="aac",
                   audio_bitrate="192k", use_source_fps=True, prompt=None, extra_pnginfo=None):
        
        input_path = video['path']
        output_dir = folder_paths.get_output_directory()
        
        # Use folder_paths for sequential naming
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        
        # Determine output extension based on codec
        ext_map = {
            'libx264': 'mp4',
            'libx265': 'mp4',
            'mpeg4': 'avi',
            'libvpx-vp9': 'webm'
        }
        ext = ext_map.get(codec, 'mp4')
        
        output_filename = f"{filename}_{counter:05}_.{ext}"
        output_path = os.path.join(full_output_folder, output_filename)
        
        # Use source FPS or specified FPS
        actual_fps = video['fps'] if use_source_fps else fps
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', codec,
            '-preset', preset,
            '-crf', str(quality_crf),
            '-pix_fmt', pixel_format,
            '-r', str(actual_fps),
        ]
        
        # Audio settings
        if video['has_audio']:
            cmd.extend(['-c:a', audio_codec, '-b:a', audio_bitrate])
        else:
            cmd.extend(['-an'])  # No audio
        
        # VP9 specific settings
        if codec == 'libvpx-vp9':
            # VP9 uses different quality setting
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', codec,
                '-crf', str(quality_crf), '-b:v', '0',
                '-pix_fmt', pixel_format,
                '-r', str(actual_fps),
            ]
            if video['has_audio']:
                cmd.extend(['-c:a', 'libopus', '-b:a', audio_bitrate])
        
        cmd.append(output_path)
        
        # Run ffmpeg
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[RizzNodes VideoSuit] Save error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video save timeout - file may be too large")
        
        # Get output video info
        output_video = get_video_info(output_path)
        
        return (output_path, output_video)


# ============================================================================
# Node 5: RizzPreviewVideo
# ============================================================================

class RizzPreviewVideo:
    """Preview video in ComfyUI web interface (similar to audio preview)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "autoplay": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": False}),
                "max_size": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Max dimension (0 = original size)"}),
                "pixel_perfect": ("BOOLEAN", {"default": False, "tooltip": "Use nearest-neighbor scaling for sharp edges (good for pixel art)."}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "RizzNodes/Video"
    
    def preview(self, video, autoplay=True, loop=False, max_size=0, pixel_perfect=False):
        import shutil
        import random
        
        input_path = video['path']
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename for preview
        random_suffix = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        preview_filename = f"ComfyUI_temp_{random_suffix}_.mp4"
        preview_path = os.path.join(temp_dir, preview_filename)
        
        # If max_size is set, resize the video for preview
        if max_size > 0 and (video['width'] > max_size or video['height'] > max_size):
            # Scale down for preview
            scale_flags = ":flags=neighbor" if pixel_perfect else ""
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease{scale_flags}",
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
                '-c:a', 'aac', '-b:a', '128k',
                preview_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=120)
            except:
                # Fallback to copy
                shutil.copy2(input_path, preview_path)
        else:
            # Copy the video to temp directory for web serving
            # If already mp4, just copy; otherwise transcode to mp4 for browser compatibility
            if input_path.lower().endswith('.mp4'):
                shutil.copy2(input_path, preview_path)
            else:
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    preview_path
                ]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=120)
                except:
                    shutil.copy2(input_path, preview_path)
        
        # Return UI data in the format ComfyUI expects for video preview
        # This matches the PreviewVideo format: {"images": [...], "animated": (True,)}
        return {
            "ui": {
                "images": [{
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp"
                }],
                "animated": (True,)
            }
        }


        return (output_path, output_video)


# ============================================================================
# Node 6: RizzSeparateVideoAudio
# ============================================================================

class RizzSeparateVideoAudio:
    """Separates video and audio tracks from a VIDEO input."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "separate"
    CATEGORY = "RizzNodes/Video"
    
    def separate(self, video):
        # Extract audio and also create a muted video file
        
        # Check if video has audio
        if not video.get('has_audio', False):
             # Return as-is
             empty_audio = {"waveform": torch.zeros((1, 1), dtype=torch.float32), "sample_rate": 48000}
             return (video, empty_audio)
             
        input_path = video['path']
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Paths
        audio_path = os.path.join(temp_dir, f"extracted_audio_{uuid.uuid4().hex}.wav")
        muted_video_path = os.path.join(temp_dir, f"muted_video_{uuid.uuid4().hex}.mp4")
        
        try:
            # 1. Extract Audio
            cmd_audio = [
                'ffmpeg', '-y', '-i', input_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                audio_path
            ]
            subprocess.run(cmd_audio, check=True, capture_output=True)
            
            # Read wav for AUDIO output
            import soundfile as sf
            waveform, sample_rate = sf.read(audio_path)
            waveform = torch.from_numpy(waveform).float()
            
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(1) # [T, 1]
            
            # Transpose to [C, T] -> Unsqueeze batch -> [1, C, T]
            waveform = waveform.t().unsqueeze(0) 
            audio_output = {"waveform": waveform, "sample_rate": sample_rate}
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

            # 2. Create Muted Video (Copy video stream, remove audio)
            cmd_video = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'copy', '-an',
                muted_video_path
            ]
            subprocess.run(cmd_video, check=True, capture_output=True)
            
            # Get info for new video
            muted_video_info = get_video_info(muted_video_path)
            # Ensure metadata says no audio
            muted_video_info['has_audio'] = False
                
            return (muted_video_info, audio_output)
            
        except subprocess.CalledProcessError as e:
            print(f"[RizzNodes VideoSuit] FFmpeg error during separation: {e}")
            if e.stderr:
                print(f"Stderr: {e.stderr.decode('utf-8')}")
            empty_audio = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
            return (video, empty_audio)
        except Exception as e:
            print(f"[RizzNodes VideoSuit] Error during separation: {e}")
            empty_audio = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
            return (video, empty_audio)


# ============================================================================
# Node 7: RizzExtractAllFrames
# ============================================================================

class RizzExtractAllFrames:
    """Extract all frames from a video as a batch of images.
    WARNING: Can consume huge amounts of RAM!"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "limit_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Limit number of frames to extract. 0 = unlimited."}),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "Start extraction at this time (seconds)."}),
                "end_time": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "End extraction at this time (seconds). 0 = end of video."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "extract_all"
    CATEGORY = "RizzNodes/Video"
    
    def extract_all(self, video, limit_frames=0, start_time=0.0, end_time=0.0):
        video_path = video['path']
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
             raise ValueError(f"Could not open video: {video_path}")
        
        fps = video.get('fps', 24.0)
        total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        
        if end_time > 0:
            end_frame = min(int(end_time * fps), total_frames_in_file)
        else:
            end_frame = total_frames_in_file
            
        # Ensure range is valid
        if start_frame >= end_frame:
             return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
             
        max_frames_to_read = end_frame - start_frame
        
        # Apply strict limit if set
        if limit_frames > 0:
            max_frames_to_read = min(max_frames_to_read, limit_frames)
            
        # Pre-allocate output tensor: [B, H, W, C]
        # Note: BGR -> RGB will be done during assignment
        print(f"[RizzNodes] Pre-allocating {max_frames_to_read} frames ({width}x{height})...")
        try:
            batch = torch.empty((max_frames_to_read, height, width, 3), dtype=torch.float32)
        except RuntimeError:
            raise RuntimeError(f"Not enough RAM to allocate {max_frames_to_read} frames! Try reducing limit_frames.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_read = 0
        
        print(f"[RizzNodes] Extracting frames {start_frame} to {start_frame + max_frames_to_read}...")
        
        for i in range(max_frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to 0-1 and assign directly to pre-allocated tensor
            batch[i] = torch.from_numpy(frame.astype(np.float32) / 255.0)
            frames_read += 1
            
        cap.release()
        
        # If we read fewer frames than expected (e.g. EOF), slice the tensor
        if frames_read < max_frames_to_read:
            print(f"[RizzNodes] Read {frames_read} frames (expected {max_frames_to_read}). trimming...")
            batch = batch[:frames_read]
            
        if frames_read == 0:
             return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        print(f"[RizzNodes] Output batch shape: {batch.shape}")
        
        import gc
        gc.collect()
        
        return (batch,)


# ============================================================================
# Node 8: RizzEditClips
# ============================================================================

MAX_VIDEO_SLOTS = 25

TRANSITION_MODES = [
    "None", "fade", "wipeleft", "wiperight", "wipeup", "wipedown", 
    "slideleft", "slideright", "slideup", "slidedown", "circlecrop", 
    "rectcrop", "distance", "fadeblack", "fadewhite", "radial", 
    "smoothleft", "smoothright", "smoothup", "smoothdown", 
    "circleopen", "circleclose", "vertopen", "vertclose", 
    "horzopen", "horzclose", "dissolve", "pixelize", "diagtl", 
    "diagtr", "diagbl", "diagbr", "hlslice", "hrslice", 
    "vuslice", "vdslice", "hblur", "fadegrayscale", 
    "wipetl", "wipetr", "wipebl", "wipebr", "squeezeh", "squeezev"
]

class RizzEditClips:
    """Combine and trim video clips with transitions.
    - If 1 clip: trim start/end.
    - If multiple: trim start/end, then combine with optional transitions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "video_count": ("INT", {
                    "default": 1, "min": 1, "max": MAX_VIDEO_SLOTS, "step": 1,
                    "tooltip": "Number of video clips to combine."
                }),
                "trim_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "Seconds to trim from the START of the 1st clip."
                }),
                "trim_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "Seconds to trim from the END of the LAST clip."
                }),
                "process_clips": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable frame interpolation and advanced processing for smooth framerate conversion. (Slower)"
                }),
            },
            "optional": {}
        }
        
        # Dynamic inputs
        for i in range(1, MAX_VIDEO_SLOTS + 1):
             inputs["optional"][f"video_{i}"] = ("VIDEO",)
             if i > 1:
                 # Transition FROM previous clip TO this clip
                 inputs["optional"][f"transition_{i}"] = (TRANSITION_MODES, {"default": "None"})
                 inputs["optional"][f"trans_len_{i}"] = ("FLOAT", {
                     "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1,
                     "tooltip": "Duration of transition in seconds."
                 })
             
        return inputs
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "edit_clips"
    CATEGORY = "RizzNodes/Video"
    
    def edit_clips(self, video_count, trim_start=0.0, trim_end=0.0, process_clips=False, **kwargs):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"rizz_edit_{uuid.uuid4().hex}.mp4")
        
        # Collect video inputs
        video_inputs = []
        for i in range(1, video_count + 1):
            vid = kwargs.get(f"video_{i}")
            if vid:
                video_inputs.append({
                    'video': vid,
                    'index': i,
                    'transition': kwargs.get(f"transition_{i}", "None"),
                    'trans_len': kwargs.get(f"trans_len_{i}", 1.0)
                })
                
        if not video_inputs:
            raise ValueError("No video inputs provided.")
            
        # Determine common properties (use first video as reference)
        ref_video = video_inputs[0]['video']
        width = ref_video['width']
        height = ref_video['height']
        fps = ref_video.get('fps', 24.0)
        
        # Check global audio presence
        has_audio = any(v['video'].get('has_audio', False) for v in video_inputs)
        
        cmd = ['ffmpeg', '-y']
        
        # Add inputs
        for v in video_inputs:
            cmd.extend(['-i', v['video']['path']])
            
        filter_complex = []
        
        # We need to process each segment first (trim/scale)
        # Then we chain them:
        # v0 + v1 -> m1 (xfade)
        # m1 + v2 -> m2 (xfade)
        # ...
        
        # Segment preparation
        seg_v_labels = []
        seg_a_labels = []
        seg_durations = []
        
        for idx, item in enumerate(video_inputs):
            vid = item['video']
            in_v = f"{idx}:v"
            in_a = f"{idx}:a"
            
            out_v = f"segv{idx}"
            out_a = f"sega{idx}"
            
            # Trim logic
            start_cut = 0.0
            
            duration = vid['duration']
            
            if idx == 0:
                start_cut = trim_start
                
            if idx == len(video_inputs) - 1:
                # Calculate end timestamp
                end_time_abs = duration - trim_end
                if end_time_abs <= start_cut:
                    end_time_abs = start_cut + 0.1
                target_end = end_time_abs
            else:
                target_end = duration
                
            # Filter Chain for Segment
            filters_chain = []
            
            # 1. Scale and Normalize FPS/Timebase
            # xfade requires matching height/width AND timebase/framerate
            # We enforce standard TB and the FPS of the first clip
            filters_chain.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
            
            # FPS Handling:
            # If process_clips is True:
            #   If src_fps < target_fps: Interpolate (Frame Blend)
            #   If src_fps > target_fps: Downscale (Drop frames)
            # Else (False):
            #   Just force FPS (Drop/Dup frames) - fast and simple.
            
            src_fps = vid.get('fps', fps)
            if process_clips:
                if src_fps < fps:
                    print(f"[RizzNodes] Interpolating clip {idx+1} from {src_fps}fps to {fps}fps (Frame Blend)")
                    filters_chain.append(f"minterpolate='mi_mode=blend:fps={fps}'")
                elif src_fps > fps:
                    print(f"[RizzNodes] Downscaling clip {idx+1} from {src_fps}fps to {fps}fps")
                    filters_chain.append(f"fps={fps}")
                else:
                    filters_chain.append(f"fps={fps}")
            else:
                 # Default fast behavior: just force fps/timebase without blend
                 filters_chain.append(f"fps={fps}")
                
            filters_chain.append("format=yuv420p") # xfade also likes matching pixel formats
            
            # 2. Trim
            trim_args = []
            if start_cut > 0:
                trim_args.append(f"start={start_cut}")
            if idx == len(video_inputs) - 1 and trim_end > 0:
                 trim_args.append(f"end={target_end}")
            
            if trim_args:
                filters_chain.append(f"trim={':'.join(trim_args)}")
            
            # 3. SETPTS (Must be settb=AVTB to match for xfade?) 
            # Actually settb=AVTB helps, but fps filter usually sets tb to 1/fps
            # Let's add settb=AVTB just to be safe if fps doesn't
            filters_chain.append("settb=AVTB")
            filters_chain.append("setpts=PTS-STARTPTS")
            
            filter_complex.append(f"[{in_v}]{','.join(filters_chain)}[{out_v}]")
            seg_v_labels.append(f"[{out_v}]")
            
            # Calculate segment duration
            actual_duration = target_end - start_cut
            seg_durations.append(actual_duration)
            
            # 4. Audio
            if has_audio:
                if vid.get('has_audio', False):
                    atrim_args = []
                    if start_cut > 0:
                        atrim_args.append(f"start={start_cut}")
                    if idx == len(video_inputs) - 1 and trim_end > 0:
                        atrim_args.append(f"end={target_end}")
                    
                    afilter = []
                    if atrim_args:
                        afilter.append(f"atrim={':'.join(atrim_args)}")
                    afilter.append("asetpts=PTS-STARTPTS")
                    
                    filter_complex.append(f"[{in_a}]{','.join(afilter)}[{out_a}]")
                    seg_a_labels.append(f"[{out_a}]")
                else:
                    # Generate silence
                    if actual_duration < 0.1: actual_duration = 0.1
                    silence_label = f"silence{idx}"
                    filter_complex.append(f"anullsrc=channel_layout=stereo:sample_rate=44100,atrim=duration={actual_duration}[{silence_label}]")
                    seg_a_labels.append(f"[{silence_label}]")

        # Now, chain them together
        # We start with seg 0
        current_v = seg_v_labels[0]
        current_a = seg_a_labels[0] if has_audio else None
        
        accumulated_duration = seg_durations[0]
        
        for i in range(1, len(video_inputs)):
            next_v = seg_v_labels[i]
            next_a = seg_a_labels[i] if has_audio else None
            next_dur = seg_durations[i]
            
            item = video_inputs[i]
            trans_mode = item['transition']
            trans_len = item['trans_len']
            
            # Validate max transition length vs accumulated duration and next clip duration
            # Offset = accumulated_duration - trans_len
            # We need accumulated_duration > trans_len AND next_dur > trans_len
            # If not, fallback to simple concat (offset = accumulated_duration, len=0)
            
            if trans_mode != "None" and trans_len > 0 and accumulated_duration > trans_len and next_dur > trans_len:
                # XFADE
                offset = accumulated_duration - trans_len
                out_mix_v = f"mixv{i}"
                
                filter_complex.append(f"{current_v}{next_v}xfade=transition={trans_mode}:duration={trans_len}:offset={offset}[{out_mix_v}]")
                current_v = f"[{out_mix_v}]"
                
                # ACROSSFADE for Audio
                if has_audio:
                    out_mix_a = f"mixa{i}"
                    # Audio crossfade doesn't use offset syntax exactly same as xfade usually?
                    # actually 'acrossfade' filter: "d=DURATION:c1=curve1:c2=curve2"
                    # It automatically overlaps by duration. It consumes streams.
                    # Wait, xfade uses absolute offset. acrossfade overlaps the end of 1st with start of 2nd.
                    # xfade offset logic basically assumes streams are playing from 0?
                    # When we use complex filter chains, intermediate streams start at 0?
                    # YES.
                    # IMPORTANT: xfade takes two streams.
                    # Offset is relative to the start of the first stream.
                    
                    # acrossfade usage: params 'd' (duration), 'o' (overlap - bool?), 'c1', 'c2'. Usually just d=...
                    # But acrossfade does NOT take an offset. It just overlaps end of A and start of B.
                    # This matches xfade logic nicely properly if we just want them to overlap.
                    # However, does xfade change the PTS?
                    # xfade output duration = durA + durB - trans_len
                    # acrossfade output duration = durA + durB - trans_len
                    
                    filter_complex.append(f"{current_a}{next_a}acrossfade=d={trans_len}[{out_mix_a}]")
                    current_a = f"[{out_mix_a}]"
                
                # Update accumulated duration
                accumulated_duration = accumulated_duration + next_dur - trans_len
                
            else:
                # CONCAT (No transition)
                # If we mix xfade and concat, we must use concat filter for [current] + [next].
                # concat filter takes n=2
                
                out_concat_v = f"concatv{i}"
                filter_complex.append(f"{current_v}{next_v}concat=n=2:v=1:a=0[{out_concat_v}]")
                current_v = f"[{out_concat_v}]"
                
                if has_audio:
                    out_concat_a = f"concata{i}"
                    filter_complex.append(f"{current_a}{next_a}concat=n=2:v=0:a=1[{out_concat_a}]")
                    current_a = f"[{out_concat_a}]"
                    
                accumulated_duration += next_dur

        # Final mapping
        cmd.extend(['-filter_complex', ";".join(filter_complex)])
        cmd.extend(['-map', current_v])
        if has_audio and current_a:
             cmd.extend(['-map', current_a])
             
        # Encoding output
        cmd.extend([
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p',
        ])
        if has_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
            
        cmd.append(output_path)
        
        # Run
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[RizzNodes] EditClips FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg Error: {result.stderr[:500]}")
                
            return (get_video_info(output_path),)
            
        except Exception as e:
            print(f"[RizzNodes] Error in EditClips (Transition): {e}")
            raise e


# ============================================================================
# Node Mappings
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "RizzLoadVideo": RizzLoadVideo,
    "RizzLoadAudio": RizzLoadAudio,
    "RizzExtractFrames": RizzExtractFrames,
    "RizzVideoEffects": RizzVideoEffects,
    "RizzSaveVideo": RizzSaveVideo,
    "RizzPreviewVideo": RizzPreviewVideo,
    "RizzSeparateVideoAudio": RizzSeparateVideoAudio,
    "RizzExtractAllFrames": RizzExtractAllFrames,
    "RizzEditClips": RizzEditClips,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RizzLoadVideo": "🎥 Load Video (Rizz)",
    "RizzLoadAudio": "🎵 Load Audio (Rizz)",
    "RizzExtractFrames": "🎞️ Extract Frames (Start/End)",
    "RizzVideoEffects": "🎬 Video Effects (Rizz)",
    "RizzSaveVideo": "💾 Save Video (Rizz)",
    "RizzPreviewVideo": "📺 Preview Video (Rizz)",
    "RizzSeparateVideoAudio": "🔊 Separate Video/Audio",
    "RizzExtractAllFrames": "🎞️ Extract ALL Frames (Batch)",
    "RizzEditClips": "✂️ Edit & Combine Clips (Rizz)",
}

