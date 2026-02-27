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

def ensure_video_dict(video):
    """Ensure the video input is a dictionary. If it's a VideoInput object (like VideoFromComponents),
    save it to a temporary file and return the metadata dictionary."""
    if isinstance(video, dict) and 'path' in video:
        return video
    
    # Handle new ComfyAPI VideoInput objects (e.g., VideoFromComponents, VideoFromFile)
    if video is not None and hasattr(video, 'save_to'):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"rizz_vinput_{uuid.uuid4().hex}.mp4")
        
        try:
            # We use .save_to() specifically for VideoInput subclasses in ComfyUI Core/API
            print(f"[RizzNodes VideoSuit] Processing video input object of type: {type(video).__name__}")
            video.save_to(temp_path)
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                return get_video_info(temp_path)
            else:
                print(f"[RizzNodes VideoSuit] Failed to save video object to {temp_path}")
        except Exception as e:
            print(f"[RizzNodes VideoSuit] Error converting video object: {e}")
            if isinstance(video, dict): return video 
            
    # Fallback to dictionary checking if it's already a dict but maybe missing 'path'
    if isinstance(video, dict):
        if 'path' in video: return video
        # Maybe it's a different dict format?
        return video
        
    # Final check: if it's a list, we might have a batch. 
    # Current RizzNodes don't support batches of VideoInput objects directly yet.
    if isinstance(video, (list, tuple)) and len(video) > 0:
        return ensure_video_dict(video[0])

    raise TypeError(f"Expected VIDEO (dict) or VideoInput object, got {type(video).__name__}. This node requires a compatible video format (dictionary with 'path' or ComfyUI VideoInput).")


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
            if folder_path == "RizzVideo":
                path = os.path.join(folder_paths.get_output_directory(), "RizzVideo", file)
            else:
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
            if folder_path == "RizzAudio":
                path = os.path.join(folder_paths.get_output_directory(), "RizzAudio", file)
            else:
                path = os.path.join(folder_path, file)
            if os.path.exists(path):
                return os.path.getmtime(path)
        return float("nan")

    def load_audio(self, folder_path, file):
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
        video = ensure_video_dict(video)
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
        video = ensure_video_dict(video)
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
                "quality_crf": ("INT", {"default": 23, "min": 0, "max": 51, "tooltip": "Quality control: CRF for x264/x265/VP9, mapped to qscale for MPEG-4. Lower = better quality, larger file."}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "medium", "tooltip": "Encoding speed preset for x264/x265. Slower = better compression but longer encoding time."}),
                "pixel_format": (["yuv420p", "yuv444p"], {"default": "yuv420p", "tooltip": "yuv420p is widely compatible. yuv444p preserves more color detail but less compatible."}),
                "audio_codec": (["aac", "mp3"], {"default": "aac", "tooltip": "Audio codec for MP4 outputs. VP9/WebM always uses Opus for compatibility."}),
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
        
        video = ensure_video_dict(video)
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
            'mpeg4': 'mp4',
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
            '-pix_fmt', pixel_format,
            '-r', str(actual_fps),
        ]

        # Codec-specific quality options
        if codec in ['libx264', 'libx265']:
            cmd.extend(['-preset', preset, '-crf', str(quality_crf)])
        elif codec == 'libvpx-vp9':
            cmd.extend(['-crf', str(quality_crf), '-b:v', '0'])
        elif codec == 'mpeg4':
            # MPEG-4 does not use CRF/preset; map quality_crf 0..51 -> qscale 1..31
            mpeg4_q = max(1, min(31, int(round(1 + (quality_crf / 51.0) * 30))))
            cmd.extend(['-q:v', str(mpeg4_q)])
        
        # Audio settings
        if video['has_audio']:
            if codec == 'libvpx-vp9':
                cmd.extend(['-c:a', 'libopus', '-b:a', audio_bitrate])
            else:
                cmd.extend(['-c:a', audio_codec, '-b:a', audio_bitrate])
        else:
            cmd.extend(['-an'])  # No audio
        
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
        
        video = ensure_video_dict(video)
        input_path = video['path']
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename for preview
        random_suffix = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        preview_filename = f"ComfyUI_temp_{random_suffix}_.mp4"
        preview_path = os.path.join(temp_dir, preview_filename)

        def _preview_ready(path):
            return os.path.exists(path) and os.path.getsize(path) > 0
        
        # Always transcode preview to browser-safe H.264/AAC.
        # Some MP4 inputs (e.g. x265 or mpeg4 codec) won't play in browser players.
        vf_filters = []
        if max_size > 0 and (video['width'] > max_size or video['height'] > max_size):
            scale_flags = ":flags=neighbor" if pixel_perfect else ""
            vf_filters.append(
                f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease{scale_flags}"
            )
        vf_filters.append("format=yuv420p")

        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', ",".join(vf_filters),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        ]
        if video.get('has_audio', False):
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        else:
            cmd.extend(['-an'])
        cmd.append(preview_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[:500] if result.stderr else "ffmpeg preview transcode failed")
        except Exception as e:
            print(f"[RizzNodes VideoSuit] Preview transcode fallback: {e}")
            # Fallback to copy so at least the file remains accessible for download.
            shutil.copy2(input_path, preview_path)

        # Last-chance fallback in case ffmpeg/copy produced no valid file
        if not _preview_ready(preview_path):
            shutil.copy2(input_path, preview_path)
            if not _preview_ready(preview_path):
                raise RuntimeError("Failed to prepare video preview file.")
        
        # The custom frontend handler reads `videos`/`video`.
        preview_entry = {
            "filename": preview_filename,
            "subfolder": "",
            "type": "temp",
            "format": "video/mp4",
            "autoplay": autoplay,
            "loop": loop,
        }
        return {
            "ui": {
                "videos": [preview_entry],
                "video": [preview_entry],
            }
        }


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
        video = ensure_video_dict(video)
        
        # Check if video has audio
        if not video.get('has_audio', False):
             # Return as-is
             empty_audio = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
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
        video = ensure_video_dict(video)
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
# Node 8: RizzFramesToVideoBatch
# ============================================================================

class RizzFramesToVideoBatch:
    """Convert a batch of IMAGE frames into a VIDEO object for downstream nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Frames per second for the created video."}),
            },
            "optional": {
                "limit_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Limit number of frames to encode. 0 = encode all frames."}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "frames_to_video"
    CATEGORY = "RizzNodes/Video"

    def frames_to_video(self, images, fps, limit_frames=0):
        import shutil

        if images is None:
            raise ValueError("No input images provided.")

        # Ensure BHWC tensor
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if len(images.shape) != 4:
            raise ValueError(f"Expected IMAGE batch in BHWC format, got shape: {tuple(images.shape)}")

        total_frames = int(images.shape[0])
        if total_frames <= 0:
            raise ValueError("Input image batch is empty.")

        frame_count = min(total_frames, int(limit_frames)) if limit_frames and limit_frames > 0 else total_frames
        if frame_count <= 0:
            raise ValueError("No frames selected for encoding.")

        height = int(images.shape[1])
        width = int(images.shape[2])

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        frames_dir = tempfile.mkdtemp(prefix="rizz_frames_to_video_", dir=temp_dir)
        output_path = os.path.join(temp_dir, f"rizz_frames_to_video_{uuid.uuid4().hex}.mp4")

        try:
            # Write frame sequence for ffmpeg input
            for i in range(frame_count):
                frame = images[i]
                frame_np = np.clip(frame.detach().cpu().numpy(), 0.0, 1.0)

                # Normalize to RGB for video encoding
                if frame_np.ndim == 2:
                    frame_np = np.stack([frame_np, frame_np, frame_np], axis=-1)
                elif frame_np.shape[-1] == 1:
                    frame_np = np.repeat(frame_np, 3, axis=-1)
                elif frame_np.shape[-1] >= 3:
                    frame_np = frame_np[..., :3]
                else:
                    raise ValueError(f"Unsupported channel count in frame: {frame_np.shape}")

                frame_u8 = (frame_np * 255.0).astype(np.uint8)
                frame_img = Image.fromarray(frame_u8, mode="RGB")
                frame_img.save(os.path.join(frames_dir, f"frame_{i:06d}.png"))

            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-start_number', '0',
                '-i', os.path.join(frames_dir, 'frame_%06d.png'),
            ]

            # yuv420p requires even dimensions; pad if needed.
            if width % 2 != 0 or height % 2 != 0:
                cmd.extend(['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])

            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-an',
                output_path
            ])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[RizzNodes VideoSuit] FramesToVideo error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")

            output_video = get_video_info(output_path)
            output_video['has_audio'] = False
            return (output_video,)

        finally:
            # Clean temporary frame sequence
            shutil.rmtree(frames_dir, ignore_errors=True)


# ============================================================================
# Node 9: RizzEditClips
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
                "zoom_in": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Zoom factor into the center of the video. 1.0 = no zoom, 2.0 = 2x zoom, etc."
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
    
    def edit_clips(self, video_count, trim_start=0.0, trim_end=0.0, process_clips=False, zoom_in=1.0, **kwargs):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"rizz_edit_{uuid.uuid4().hex}.mp4")
        
        # Collect video inputs
        video_inputs = []
        for i in range(1, video_count + 1):
            vid = kwargs.get(f"video_{i}")
            if vid:
                vid = ensure_video_dict(vid)
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

        if zoom_in > 1.0:
            crop_w = int(width / zoom_in)
            crop_h = int(height / zoom_in)
            if crop_w % 2 != 0:
                crop_w -= 1
            if crop_h % 2 != 0:
                crop_h -= 1
            crop_x = (width - crop_w) // 2
            crop_y = (height - crop_h) // 2
            zoom_filter = f"{current_v}crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={width}:{height}:flags=lanczos[zoomed]"
            filter_complex.append(zoom_filter)
            current_v = "[zoomed]"

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
# Node 10: RizzTimelineEditor
# ============================================================================

class RizzTimelineEditor:
    """Timeline renderer with 3 video and 3 audio sources.
    The interactive editor lives in js/rizz_timeline_editor.js and writes timeline_json.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_json": ("STRING", {
                    "default": "{\"timeline_length\":10.0,\"playhead\":0.0,\"video_clips\":[],\"audio_clips\":[]}",
                    "multiline": True,
                    "tooltip": "Internal timeline state (managed by the timeline editor UI).",
                }),
                "output_width": ("INT", {
                    "default": 1280, "min": 64, "max": 8192, "step": 2,
                    "tooltip": "Output video width in pixels.",
                }),
                "output_height": ("INT", {
                    "default": 720, "min": 64, "max": 8192, "step": 2,
                    "tooltip": "Output video height in pixels.",
                }),
                "output_fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1,
                    "tooltip": "Output framerate.",
                }),
                "include_video_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include audio from placed video clips if present.",
                }),
                "end_with_last_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, output extends to include the last placed clip. If false, output uses timeline length and clips past it are cut.",
                }),
                "master_volume": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Final mix output gain.",
                }),
            },
            "optional": {
                "video_1": ("VIDEO",),
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "render_timeline"
    CATEGORY = "RizzNodes/Video"

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _to_bool(value, default=True):
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off"}
        return bool(value)

    @staticmethod
    def _fmt_num(value):
        value = max(float(value), 0.0)
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _parse_timeline(self, timeline_json):
        parsed = {
            "timeline_length": 0.0,
            "playhead": 0.0,
            "video_clips": [],
            "audio_clips": [],
        }

        if not timeline_json:
            return parsed

        try:
            data = json.loads(timeline_json)
        except Exception:
            return parsed

        if not isinstance(data, dict):
            return parsed

        parsed["timeline_length"] = max(0.0, self._safe_float(data.get("timeline_length", 0.0), 0.0))
        parsed["playhead"] = max(0.0, self._safe_float(data.get("playhead", 0.0), 0.0))

        def _normalize_clips(raw_list, media):
            out = []
            if not isinstance(raw_list, list):
                return out

            def _normalize_transition(raw_t):
                if not isinstance(raw_t, dict):
                    return None
                t_type = str(raw_t.get("type", "None")).strip()
                t_dur = max(0.0, self._safe_float(raw_t.get("duration", 0.0), 0.0))
                if t_type == "None" or t_dur <= 0.0:
                    return None
                # Keep transition types explicit to avoid unexpected ffmpeg chains.
                if t_type not in {
                    "Fade", "Smooth", "Dissolve", "Dip Black", "Dip White",
                    "Slide Left", "Slide Right", "Wipe Left", "Wipe Right",
                    "Zoom In", "Zoom Out"
                }:
                    return None
                return {"type": t_type, "duration": t_dur}

            for idx, raw in enumerate(raw_list):
                if not isinstance(raw, dict):
                    continue

                src = int(self._safe_float(raw.get("src", 1), 1))
                if src < 1 or src > 3:
                    continue

                track = int(self._safe_float(raw.get("track", 0), 0))
                track = max(0, min(2, track))

                start = max(0.0, self._safe_float(raw.get("start", 0.0), 0.0))
                trim_in = max(0.0, self._safe_float(raw.get("in", 0.0), 0.0))
                dur = max(0.05, self._safe_float(raw.get("dur", 1.0), 1.0))
                volume = max(0.0, self._safe_float(raw.get("volume", 1.0), 1.0))
                enabled = self._to_bool(raw.get("enabled", True), True)
                if not enabled:
                    continue

                clip_id = str(raw.get("id", f"{media}_{idx+1}"))
                out.append({
                    "id": clip_id,
                    "src": src,
                    "track": track,
                    "start": start,
                    "in": trim_in,
                    "dur": dur,
                    "volume": volume,
                    "transition_in": _normalize_transition(raw.get("transition_in")),
                    "transition_out": _normalize_transition(raw.get("transition_out")),
                })
            return out

        parsed["video_clips"] = _normalize_clips(data.get("video_clips", []), "video")
        parsed["audio_clips"] = _normalize_clips(data.get("audio_clips", []), "audio")
        return parsed

    def _audio_to_wav(self, audio_data, temp_dir, stem):
        if not isinstance(audio_data, dict):
            return None

        waveform = audio_data.get("waveform")
        sample_rate = int(self._safe_float(audio_data.get("sample_rate", 44100), 44100))
        if sample_rate <= 0:
            sample_rate = 44100
        if waveform is None:
            return None

        if isinstance(waveform, torch.Tensor):
            wave = waveform.detach().cpu().float()
        else:
            wave = torch.tensor(waveform, dtype=torch.float32)

        if wave.dim() == 3:
            wave = wave[0]
        elif wave.dim() == 1:
            wave = wave.unsqueeze(0)
        elif wave.dim() != 2:
            return None

        # Expected Comfy shape: [channels, samples]
        if wave.shape[0] > wave.shape[1] and wave.shape[1] <= 8:
            wave = wave.transpose(0, 1)

        if wave.dim() != 2 or wave.shape[-1] <= 0:
            return None

        samples = wave.transpose(0, 1).contiguous().numpy()
        samples = np.clip(samples, -1.0, 1.0)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        import soundfile as sf
        audio_path = os.path.join(temp_dir, f"{stem}.wav")
        sf.write(audio_path, samples, sample_rate, subtype="PCM_16")

        duration = float(samples.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0
        return {
            "path": audio_path,
            "duration": max(0.0, duration),
            "sample_rate": sample_rate,
        }

    def render_timeline(self, timeline_json, output_width=1280, output_height=720, output_fps=30.0,
                        include_video_audio=True, end_with_last_clip=True, master_volume=1.0, **kwargs):
        output_width = int(max(64, output_width))
        output_height = int(max(64, output_height))
        output_fps = max(1.0, float(output_fps))
        master_volume = max(0.0, float(master_volume))

        timeline = self._parse_timeline(timeline_json)

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"rizz_timeline_{uuid.uuid4().hex}.mp4")

        temp_audio_files = []

        try:
            video_sources = {}
            for i in range(1, 4):
                video_input = kwargs.get(f"video_{i}")
                if video_input is None:
                    continue
                try:
                    video_sources[i] = ensure_video_dict(video_input)
                except Exception as e:
                    print(f"[RizzNodes Timeline] Failed to read video_{i}: {e}")

            audio_sources = {}
            for i in range(1, 4):
                audio_input = kwargs.get(f"audio_{i}")
                if audio_input is None:
                    continue
                try:
                    audio_info = self._audio_to_wav(audio_input, temp_dir, f"rizz_timeline_audio_{i}_{uuid.uuid4().hex}")
                    if audio_info and os.path.exists(audio_info["path"]):
                        audio_sources[i] = audio_info
                        temp_audio_files.append(audio_info["path"])
                except Exception as e:
                    print(f"[RizzNodes Timeline] Failed to read audio_{i}: {e}")

            video_clips = []
            for clip in timeline["video_clips"]:
                source = video_sources.get(clip["src"])
                if not source:
                    continue

                source_dur = max(0.0, self._safe_float(source.get("duration", 0.0), 0.0))
                clip_in = clip["in"]
                clip_dur = clip["dur"]
                if source_dur > 0:
                    if clip_in >= source_dur:
                        continue
                    clip_dur = min(clip_dur, max(0.0, source_dur - clip_in))

                if clip_dur < 0.05:
                    continue

                t_in = clip.get("transition_in")
                t_out = clip.get("transition_out")
                max_each = max(0.0, clip_dur * 0.49)
                if isinstance(t_in, dict):
                    d = min(max_each, max(0.0, self._safe_float(t_in.get("duration", 0.0), 0.0)))
                    t_in = {"type": t_in.get("type", "Fade"), "duration": d} if d > 0.0 else None
                else:
                    t_in = None
                if isinstance(t_out, dict):
                    d = min(max_each, max(0.0, self._safe_float(t_out.get("duration", 0.0), 0.0)))
                    t_out = {"type": t_out.get("type", "Fade"), "duration": d} if d > 0.0 else None
                else:
                    t_out = None

                video_clips.append({
                    **clip,
                    "dur": clip_dur,
                    "transition_in": t_in,
                    "transition_out": t_out,
                })

            audio_clips = []
            for clip in timeline["audio_clips"]:
                source = audio_sources.get(clip["src"])
                if not source:
                    continue

                source_dur = max(0.0, self._safe_float(source.get("duration", 0.0), 0.0))
                clip_in = clip["in"]
                clip_dur = clip["dur"]
                if source_dur > 0:
                    if clip_in >= source_dur:
                        continue
                    clip_dur = min(clip_dur, max(0.0, source_dur - clip_in))

                if clip_dur < 0.05:
                    continue

                t_in = clip.get("transition_in")
                t_out = clip.get("transition_out")
                max_each = max(0.0, clip_dur * 0.49)
                if isinstance(t_in, dict):
                    d = min(max_each, max(0.0, self._safe_float(t_in.get("duration", 0.0), 0.0)))
                    t_in = {"type": t_in.get("type", "Fade"), "duration": d} if d > 0.0 else None
                else:
                    t_in = None
                if isinstance(t_out, dict):
                    d = min(max_each, max(0.0, self._safe_float(t_out.get("duration", 0.0), 0.0)))
                    t_out = {"type": t_out.get("type", "Fade"), "duration": d} if d > 0.0 else None
                else:
                    t_out = None

                audio_clips.append({
                    **clip,
                    "dur": clip_dur,
                    "transition_in": t_in,
                    "transition_out": t_out,
                })

            max_end = 0.0
            for c in video_clips:
                max_end = max(max_end, c["start"] + c["dur"])
            for c in audio_clips:
                max_end = max(max_end, c["start"] + c["dur"])

            requested_timeline_length = max(0.0, timeline["timeline_length"])
            if end_with_last_clip:
                # End exactly at the furthest clip boundary.
                # Fallback to requested timeline length when no clips are placed.
                timeline_length = max_end if max_end > 0.0 else requested_timeline_length
            else:
                timeline_length = requested_timeline_length
            if timeline_length <= 0.0:
                if video_sources:
                    timeline_length = max(self._safe_float(v.get("duration", 0.0), 0.0) for v in video_sources.values())
                elif audio_sources:
                    timeline_length = max(self._safe_float(a.get("duration", 0.0), 0.0) for a in audio_sources.values())
                else:
                    timeline_length = 5.0
            timeline_length = max(0.2, min(timeline_length, 21600.0))

            filtered_video = []
            for clip in video_clips:
                if clip["start"] >= timeline_length:
                    continue
                clip_end = min(timeline_length, clip["start"] + clip["dur"])
                clip_dur = clip_end - clip["start"]
                if clip_dur >= 0.05:
                    filtered_video.append({**clip, "dur": clip_dur})
            video_clips = filtered_video

            filtered_audio = []
            for clip in audio_clips:
                if clip["start"] >= timeline_length:
                    continue
                clip_end = min(timeline_length, clip["start"] + clip["dur"])
                clip_dur = clip_end - clip["start"]
                if clip_dur >= 0.05:
                    filtered_audio.append({**clip, "dur": clip_dur})
            audio_clips = filtered_audio

            # Composite in timeline order first so moved/split clips are not visually
            # overridden by long clips from another row purely due row index.
            # If starts are equal, higher row index is composited first and lower index
            # later (so V1 can win ties over V2/V3).
            video_clips.sort(key=lambda x: (x["start"], -x["track"], x["id"]))
            audio_clips.sort(key=lambda x: (x["start"], x["track"], x["id"]))

            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s={output_width}x{output_height}:r={self._fmt_num(output_fps)}:d={self._fmt_num(timeline_length)}",
            ]

            video_input_index = {}
            used_video_sources = sorted({c["src"] for c in video_clips})
            next_index = 1
            for src in used_video_sources:
                source_path = video_sources[src]["path"]
                cmd.extend(["-i", source_path])
                video_input_index[src] = next_index
                next_index += 1

            audio_input_index = {}
            used_audio_sources = sorted({c["src"] for c in audio_clips})
            for src in used_audio_sources:
                source_path = audio_sources[src]["path"]
                cmd.extend(["-i", source_path])
                audio_input_index[src] = next_index
                next_index += 1

            filters = []
            filters.append("[0:v]format=rgba,setpts=PTS-STARTPTS[vbase0]")
            current_v = "vbase0"
            frame_pad = 1.0 / output_fps if output_fps > 0 else 0.0

            def _slide_expr(direction, start_time, duration, is_out):
                start_s = self._fmt_num(start_time)
                dur_s = self._fmt_num(max(duration, 1e-6))
                prog = f"min(max((t-{start_s})/{dur_s},0),1)"
                if not is_out:
                    if direction == "left":
                        return f"({output_width}*(1-{prog}))"
                    return f"(-{output_width}*(1-{prog}))"
                else:
                    if direction == "left":
                        return f"(-{output_width}*{prog})"
                    return f"({output_width}*{prog})"

            for idx, clip in enumerate(video_clips):
                src_idx = video_input_index[clip["src"]]
                clip_label = f"vclip{idx}"
                out_label = f"vbase{idx+1}"

                start_s = self._fmt_num(clip["start"])
                dur_s = self._fmt_num(clip["dur"])
                in_s = self._fmt_num(clip["in"])
                frame_pad_s = self._fmt_num(frame_pad)
                v_chain = [
                    f"trim=start={in_s}:duration={dur_s}",
                    "setpts=PTS-STARTPTS",
                    f"fps={self._fmt_num(output_fps)}",
                    f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease",
                    f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "format=rgba",
                ]

                t_in = clip.get("transition_in")
                t_out = clip.get("transition_out")
                fade_like_types = {"Fade", "Smooth", "Dissolve", "Dip Black", "Dip White", "Wipe Left", "Wipe Right", "Zoom In", "Zoom Out"}
                if isinstance(t_in, dict) and t_in.get("type") in fade_like_types:
                    d = max(0.0, self._safe_float(t_in.get("duration", 0.0), 0.0))
                    if d > 0.0:
                        v_chain.append(f"fade=t=in:st=0:d={self._fmt_num(d)}:alpha=1")
                if isinstance(t_out, dict) and t_out.get("type") in fade_like_types:
                    d = max(0.0, self._safe_float(t_out.get("duration", 0.0), 0.0))
                    if d > 0.0:
                        st = max(0.0, clip["dur"] - d)
                        v_chain.append(f"fade=t=out:st={self._fmt_num(st)}:d={self._fmt_num(d)}:alpha=1")

                v_chain.append(f"tpad=stop_mode=clone:stop_duration={frame_pad_s}")
                v_chain.append(f"setpts=PTS+{start_s}/TB")

                filters.append(f"[{src_idx}:v]{','.join(v_chain)}[{clip_label}]")

                x_expr = "0"
                in_expr = None
                out_expr = None
                in_end = None
                out_start = None

                if isinstance(t_in, dict) and t_in.get("type") in {"Slide Left", "Slide Right"}:
                    d = max(0.0, self._safe_float(t_in.get("duration", 0.0), 0.0))
                    if d > 0.0:
                        direction = "left" if t_in.get("type") == "Slide Left" else "right"
                        in_expr = _slide_expr(direction, clip["start"], d, False)
                        in_end = clip["start"] + d

                if isinstance(t_out, dict) and t_out.get("type") in {"Slide Left", "Slide Right"}:
                    d = max(0.0, self._safe_float(t_out.get("duration", 0.0), 0.0))
                    if d > 0.0:
                        direction = "left" if t_out.get("type") == "Slide Left" else "right"
                        out_start = max(clip["start"], clip["start"] + clip["dur"] - d)
                        out_expr = _slide_expr(direction, out_start, d, True)

                if in_expr is not None and out_expr is not None:
                    x_expr = f"if(lt(t,{self._fmt_num(in_end)}),{in_expr},if(gte(t,{self._fmt_num(out_start)}),{out_expr},0))"
                elif in_expr is not None:
                    x_expr = f"if(lt(t,{self._fmt_num(in_end)}),{in_expr},0)"
                elif out_expr is not None:
                    x_expr = f"if(gte(t,{self._fmt_num(out_start)}),{out_expr},0)"

                filters.append(f"[{current_v}][{clip_label}]overlay=x='{x_expr}':y=0:shortest=0:eof_action=pass[{out_label}]")
                current_v = out_label

            filters.append(
                f"[{current_v}]trim=duration={self._fmt_num(timeline_length)},setpts=PTS-STARTPTS,format=yuv420p[vout]"
            )

            audio_labels = []

            if include_video_audio:
                audio_clip_idx = 0
                for clip in video_clips:
                    src_meta = video_sources.get(clip["src"], {})
                    if not src_meta.get("has_audio", False):
                        continue
                    src_idx = video_input_index[clip["src"]]
                    label = f"vca{audio_clip_idx}"
                    audio_clip_idx += 1

                    start_s = self._fmt_num(clip["start"])
                    dur_s = self._fmt_num(clip["dur"])
                    in_s = self._fmt_num(clip["in"])
                    vol_s = self._fmt_num(clip["volume"])
                    delay_ms = max(0, int(round(clip["start"] * 1000.0)))
                    a_chain = [
                        f"atrim=start={in_s}:duration={dur_s}",
                        "asetpts=PTS-STARTPTS",
                        "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo",
                    ]
                    t_in = clip.get("transition_in")
                    t_out = clip.get("transition_out")
                    if isinstance(t_in, dict):
                        d = max(0.0, self._safe_float(t_in.get("duration", 0.0), 0.0))
                        if d > 0.0:
                            a_chain.append(f"afade=t=in:st=0:d={self._fmt_num(d)}")
                    if isinstance(t_out, dict):
                        d = max(0.0, self._safe_float(t_out.get("duration", 0.0), 0.0))
                        if d > 0.0:
                            st = max(0.0, clip["dur"] - d)
                            a_chain.append(f"afade=t=out:st={self._fmt_num(st)}:d={self._fmt_num(d)}")
                    a_chain.append(f"volume={vol_s}")
                    a_chain.append(f"adelay={delay_ms}|{delay_ms}")
                    filters.append(f"[{src_idx}:a]{','.join(a_chain)}[{label}]")
                    audio_labels.append(f"[{label}]")

            for idx, clip in enumerate(audio_clips):
                src_idx = audio_input_index[clip["src"]]
                label = f"aca{idx}"

                dur_s = self._fmt_num(clip["dur"])
                in_s = self._fmt_num(clip["in"])
                vol_s = self._fmt_num(clip["volume"])
                delay_ms = max(0, int(round(clip["start"] * 1000.0)))
                a_chain = [
                    f"atrim=start={in_s}:duration={dur_s}",
                    "asetpts=PTS-STARTPTS",
                    "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo",
                ]
                t_in = clip.get("transition_in")
                t_out = clip.get("transition_out")
                if isinstance(t_in, dict):
                    d = max(0.0, self._safe_float(t_in.get("duration", 0.0), 0.0))
                    if d > 0.0:
                        a_chain.append(f"afade=t=in:st=0:d={self._fmt_num(d)}")
                if isinstance(t_out, dict):
                    d = max(0.0, self._safe_float(t_out.get("duration", 0.0), 0.0))
                    if d > 0.0:
                        st = max(0.0, clip["dur"] - d)
                        a_chain.append(f"afade=t=out:st={self._fmt_num(st)}:d={self._fmt_num(d)}")
                a_chain.append(f"volume={vol_s}")
                a_chain.append(f"adelay={delay_ms}|{delay_ms}")
                filters.append(f"[{src_idx}:a]{','.join(a_chain)}[{label}]")
                audio_labels.append(f"[{label}]")

            has_audio_output = len(audio_labels) > 0
            if has_audio_output:
                filters.append(
                    f"{''.join(audio_labels)}amix=inputs={len(audio_labels)}:duration=longest:dropout_transition=0,"
                    f"volume={self._fmt_num(master_volume)},atrim=duration={self._fmt_num(timeline_length)},"
                    f"asetpts=PTS-STARTPTS[aout]"
                )

            cmd.extend(["-filter_complex", ";".join(filters), "-map", "[vout]"])
            if has_audio_output:
                cmd.extend(["-map", "[aout]"])

            cmd.extend([
                "-r", self._fmt_num(output_fps),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
            ])

            if has_audio_output:
                cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            else:
                cmd.extend(["-an"])

            cmd.extend([
                "-movflags", "+faststart",
                "-t", self._fmt_num(timeline_length),
                output_path,
            ])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            if result.returncode != 0:
                print(f"[RizzNodes Timeline] FFmpeg error:\n{result.stderr}")
                raise RuntimeError(f"Timeline render failed: {result.stderr[:500]}")

            return (get_video_info(output_path),)
        finally:
            for p in temp_audio_files:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass


# ============================================================================
# Node 11: RizzBlurSpot
# ============================================================================

BLUR_INTERPOLATIONS = ["linear", "ease_in", "ease_out", "ease_in_out"]
BLUR_PROCESSING_MODES = ["CPU (OpenCV)", "GPU (PyTorch)", "AI (LaMa)"]
BLUR_SPOT_MODES = ["Blur", "Watermark Removal"]


def _ease_in(t):
    return t * t


def _ease_out(t):
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_in_out(t):
    return 3.0 * t * t - 2.0 * t * t * t


def _interpolate_keyframes(keyframes, frame_idx, total_frames, interp):
    if not keyframes:
        return 0.5, 0.5

    if len(keyframes) == 1:
        return keyframes[0]["x"], keyframes[0]["y"]

    if frame_idx <= keyframes[0]["frame"]:
        return keyframes[0]["x"], keyframes[0]["y"]

    if frame_idx >= keyframes[-1]["frame"]:
        return keyframes[-1]["x"], keyframes[-1]["y"]

    kf_before = keyframes[0]
    kf_after = keyframes[-1]
    for i in range(len(keyframes) - 1):
        if keyframes[i]["frame"] <= frame_idx <= keyframes[i + 1]["frame"]:
            kf_before = keyframes[i]
            kf_after = keyframes[i + 1]
            break

    span = kf_after["frame"] - kf_before["frame"]
    if span <= 0:
        return kf_before["x"], kf_before["y"]

    t = (frame_idx - kf_before["frame"]) / span

    if interp == "ease_in":
        t = _ease_in(t)
    elif interp == "ease_out":
        t = _ease_out(t)
    elif interp == "ease_in_out":
        t = _ease_in_out(t)

    x = kf_before["x"] + (kf_after["x"] - kf_before["x"]) * t
    y = kf_before["y"] + (kf_after["y"] - kf_before["y"]) * t
    return x, y


def _apply_blur_cpu(frame_np, cx, cy, radius, kernel_size):
    import cv2
    h, w = frame_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.3)
    mask = np.clip(mask, 0.0, 1.0)

    blurred = cv2.GaussianBlur(frame_np, (kernel_size, kernel_size), 0)
    mask_3d = mask[:, :, np.newaxis]
    result = (frame_np.astype(np.float32) * (1.0 - mask_3d) + blurred.astype(np.float32) * mask_3d)
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_blur_gpu(frame_tensor, cx_norm, cy_norm, radius_norm, kernel_size, device):
    from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

    _, h, w = frame_tensor.shape
    y_grid = torch.arange(h, device=device).float().unsqueeze(1).expand(h, w) / h
    x_grid = torch.arange(w, device=device).float().unsqueeze(0).expand(h, w) / w

    dist = ((x_grid - cx_norm) ** 2 + (y_grid - cy_norm) ** 2).sqrt()
    mask = torch.clamp(1.0 - dist / radius_norm, 0.0, 1.0)
    mask = mask.unsqueeze(0)

    blurred = tv_gaussian_blur(frame_tensor.unsqueeze(0), kernel_size=[kernel_size, kernel_size])[0]

    result = frame_tensor * (1.0 - mask) + blurred * mask
    return result


def _apply_inpaint_cpu(frame_np, cx, cy, radius):
    import cv2
    h, w = frame_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    inpaint_r = max(radius, 5)
    result = cv2.inpaint(frame_np, mask, inpaintRadius=inpaint_r, flags=cv2.INPAINT_NS)
    result = cv2.inpaint(result, mask, inpaintRadius=inpaint_r // 2 + 1, flags=cv2.INPAINT_NS)
    return result


def _apply_inpaint_gpu(frame_tensor, cx_norm, cy_norm, radius_norm, device):
    _, h, w = frame_tensor.shape
    y_grid = torch.arange(h, device=device).float().unsqueeze(1).expand(h, w) / h
    x_grid = torch.arange(w, device=device).float().unsqueeze(0).expand(h, w) / w

    dist = ((x_grid - cx_norm) ** 2 + (y_grid - cy_norm) ** 2).sqrt()
    hard_mask = (dist < radius_norm).float().unsqueeze(0)
    edge_mask = ((dist >= radius_norm * 0.85) & (dist < radius_norm * 1.15)).float().unsqueeze(0)
    soft_mask = torch.clamp(1.0 - dist / (radius_norm * 1.1), 0.0, 1.0).unsqueeze(0)

    from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

    edge_fill = tv_gaussian_blur(frame_tensor.unsqueeze(0), kernel_size=[5, 5])[0]
    result = frame_tensor * (1.0 - hard_mask) + edge_fill * hard_mask

    for kernel in [15, 31, 51, 71, 101, 151, 201]:
        k = min(kernel, min(h, w) - 1)
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        filled = tv_gaussian_blur(result.unsqueeze(0), kernel_size=[k, k])[0]
        result = result * (1.0 - hard_mask) + filled * hard_mask

    result = frame_tensor * (1.0 - soft_mask) + result * soft_mask
    return result


def _apply_inpaint_lama(frame_bgr, mask_uint8, lama_model, strength=1.0):
    import cv2
    from PIL import Image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    mask_pil = Image.fromarray(mask_uint8).convert("L")
    result_pil = lama_model(img_pil, mask_pil)
    result_rgb = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    if strength < 1.0:
        alpha = strength
        result_bgr = (frame_bgr.astype(np.float32) * (1.0 - alpha) + result_bgr.astype(np.float32) * alpha)
        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
    return result_bgr


class RizzBlurSpot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "mode": (BLUR_SPOT_MODES, {
                    "default": "Blur",
                    "tooltip": "Blur: Gaussian blur circle. Watermark Removal: inpaint/fill the region."
                }),
                "keyframes_json": ("STRING", {
                    "default": "[]",
                }),
                "blur_strength": ("INT", {
                    "default": 51, "min": 1, "max": 201, "step": 2,
                    "tooltip": "Blur mode: Gaussian kernel size. AI (LaMa) Watermark Removal: strength 1-100 (% blend)."
                }),
                "blur_size": ("INT", {
                    "default": 100, "min": 5, "max": 2000, "step": 1,
                    "tooltip": "Radius of the circle in pixels."
                }),
                "interpolation": (BLUR_INTERPOLATIONS, {
                    "default": "linear",
                    "tooltip": "How the position transitions between keyframes."
                }),
                "processing_mode": (BLUR_PROCESSING_MODES, {
                    "default": "CPU (OpenCV)",
                    "tooltip": "CPU uses OpenCV. GPU uses PyTorch/torchvision for faster processing."
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "apply_blur"
    CATEGORY = "RizzNodes/Video"

    def apply_blur(self, video, mode="Blur", keyframes_json="[]", blur_strength=51,
                   blur_size=100, interpolation="linear",
                   processing_mode="CPU (OpenCV)", mask=None):
        import cv2
        import shutil

        video = ensure_video_dict(video)
        video_path = video["path"]
        fps = video.get("fps", 24.0)
        width = video["width"]
        height = video["height"]

        try:
            keyframes = json.loads(keyframes_json)
            if not isinstance(keyframes, list):
                keyframes = []
        except (json.JSONDecodeError, TypeError):
            keyframes = []

        keyframes.sort(key=lambda k: k.get("frame", 0))

        kernel = blur_strength
        if kernel % 2 == 0:
            kernel += 1
        kernel = max(3, min(kernel, 201))

        use_mask = mask is not None and len(mask.shape) >= 2
        if use_mask:
            if len(mask.shape) == 4:
                mask_frames = mask
            elif len(mask.shape) == 3:
                mask_frames = mask
            else:
                mask_frames = mask.unsqueeze(0)
            print(f"[RizzNodes BlurSpot] Using connected mask ({mask_frames.shape})")
        elif not keyframes:
            print("[RizzNodes BlurSpot] No keyframes and no mask, returning original video.")
            return (video,)

        print(f"[RizzNodes BlurSpot] Mode={mode}, blur_size={blur_size}, blur_strength={kernel}, video={width}x{height}")
        if keyframes and not use_mask:
            print(f"[RizzNodes BlurSpot] Loaded {len(keyframes)} keyframe(s)")

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        frames_dir = tempfile.mkdtemp(prefix="rizz_blur_spot_", dir=temp_dir)
        output_path = os.path.join(temp_dir, f"rizz_blur_spot_{uuid.uuid4().hex}.mp4")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        use_gpu = processing_mode == "GPU (PyTorch)"
        use_lama = processing_mode == "AI (LaMa)"
        device = None
        lama_model = None

        if use_lama:
            try:
                from simple_lama_inpainting import SimpleLama
                lama_model = SimpleLama()
                print("[RizzNodes BlurSpot] AI (LaMa) model loaded")
            except ImportError:
                print("[RizzNodes BlurSpot] simple-lama-inpainting not installed. Install with: pip install simple-lama-inpainting")
                print("[RizzNodes BlurSpot] Falling back to CPU (OpenCV).")
                use_lama = False
        elif use_gpu:
            try:
                from torchvision.transforms.functional import gaussian_blur as _tv_check
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"[RizzNodes BlurSpot] GPU mode on device: {device}")
            except ImportError:
                print("[RizzNodes BlurSpot] torchvision not available, falling back to CPU.")
                use_gpu = False

        is_inpaint = mode == "Watermark Removal"
        label = "inpainting" if is_inpaint else "blurring"
        print(f"[RizzNodes BlurSpot] {label.title()} {total_frames} frames ({processing_mode})...")

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if use_mask:
                midx = min(frame_idx, mask_frames.shape[0] - 1)
                mask_np = mask_frames[midx].cpu().numpy()
                mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_LINEAR)
                mask_uint8 = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)

                if use_lama and is_inpaint:
                    lama_strength = min(max(blur_strength, 1), 100) / 100.0
                    result_bgr = _apply_inpaint_lama(frame, mask_uint8, lama_model, lama_strength)
                elif use_gpu:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(device) / 255.0
                    mask_t = torch.from_numpy(mask_np).float().to(device).unsqueeze(0)

                    if is_inpaint:
                        from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
                        hard_mask = (mask_t > 0.5).float()
                        soft_mask = mask_t
                        edge_fill = tv_gaussian_blur(frame_tensor.unsqueeze(0), kernel_size=[5, 5])[0]
                        result = frame_tensor * (1.0 - hard_mask) + edge_fill * hard_mask
                        for k in [15, 31, 51, 71, 101, 151, 201]:
                            ks = min(k, min(height, width) - 1)
                            if ks % 2 == 0: ks += 1
                            if ks < 3: ks = 3
                            filled = tv_gaussian_blur(result.unsqueeze(0), kernel_size=[ks, ks])[0]
                            result = result * (1.0 - hard_mask) + filled * hard_mask
                        result_tensor = frame_tensor * (1.0 - soft_mask) + result * soft_mask
                    else:
                        from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
                        blurred = tv_gaussian_blur(frame_tensor.unsqueeze(0), kernel_size=[kernel, kernel])[0]
                        result_tensor = frame_tensor * (1.0 - mask_t) + blurred * mask_t

                    result_np = (result_tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                else:
                    if is_inpaint:
                        inpaint_r = max(int(np.sum(mask_uint8 > 127) ** 0.5), 5)
                        result_bgr = cv2.inpaint(frame, mask_uint8, inpaintRadius=inpaint_r, flags=cv2.INPAINT_NS)
                        result_bgr = cv2.inpaint(result_bgr, mask_uint8, inpaintRadius=inpaint_r // 2 + 1, flags=cv2.INPAINT_NS)
                    else:
                        blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
                        mask_3d = mask_np[:, :, np.newaxis].astype(np.float32)
                        result_bgr = (frame.astype(np.float32) * (1.0 - mask_3d) + blurred.astype(np.float32) * mask_3d)
                        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
            else:
                x_norm, y_norm = _interpolate_keyframes(keyframes, frame_idx, total_frames, interpolation)
                cx = int(x_norm * width)
                cy = int(y_norm * height)

                if use_lama and is_inpaint:
                    circle_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.circle(circle_mask, (cx, cy), blur_size, 255, -1)
                    lama_strength = min(max(blur_strength, 1), 100) / 100.0
                    result_bgr = _apply_inpaint_lama(frame, circle_mask, lama_model, lama_strength)
                elif use_gpu:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(device) / 255.0
                    radius_norm = blur_size / max(width, height)

                    if is_inpaint:
                        result_tensor = _apply_inpaint_gpu(frame_tensor, x_norm, y_norm, radius_norm, device)
                    else:
                        result_tensor = _apply_blur_gpu(frame_tensor, x_norm, y_norm, radius_norm, kernel, device)

                    result_np = (result_tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                else:
                    if is_inpaint:
                        result_bgr = _apply_inpaint_cpu(frame, cx, cy, blur_size)
                    else:
                        result_bgr = _apply_blur_cpu(frame, cx, cy, blur_size, kernel)

            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), result_bgr)

        cap.release()

        has_audio = video.get("has_audio", False)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
        ]

        if has_audio:
            cmd.extend(["-i", video_path])

        if width % 2 != 0 or height % 2 != 0:
            cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])

        if has_audio:
            cmd.extend(["-map", "0:v", "-map", "1:a"])

        cmd.extend([
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
        ])

        if has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        else:
            cmd.extend(["-an"])

        cmd.append(output_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[RizzNodes BlurSpot] FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")

            output_video = get_video_info(output_path)
            return (output_video,)
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)


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
    "RizzFramesToVideoBatch": RizzFramesToVideoBatch,
    "RizzEditClips": RizzEditClips,
    "RizzTimelineEditor": RizzTimelineEditor,
    "RizzBlurSpot": RizzBlurSpot,
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
    "RizzFramesToVideoBatch": "🎞️ Frames to Video (Batch)",
    "RizzEditClips": "✂️ Edit & Combine Clips (Rizz)",
    "RizzTimelineEditor": "🎬 Timeline Editor (3 Video + 3 Audio)",
    "RizzBlurSpot": "🔵 Blur Spot (Keyframe)",
}
