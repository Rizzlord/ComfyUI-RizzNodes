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
    'Difference': blend_difference,
    'Exclusion': blend_exclusion,
}


# ============================================================================
# Node 1: RizzLoadVideo
# ============================================================================

class RizzLoadVideo:
    """Load a video file by path."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "placeholder": "Paste video path here",
                    "tooltip": "Absolute path to video file (e.g. /path/to/video.mp4)"
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "filename", "full_path")
    FUNCTION = "load_video"
    CATEGORY = "RizzNodes/Video"
    
    @classmethod
    def IS_CHANGED(cls, video_path=""):
        if video_path and os.path.exists(video_path):
            return os.path.getmtime(video_path)
        return float("nan")
    
    def load_video(self, video_path):
        if not video_path:
            raise ValueError("Video path is required")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_info = get_video_info(video_path)
        filename = os.path.basename(video_path)
        
        return (video_info, filename, video_path)


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
        
        # Extract first frame (at 0 seconds)
        first_frame = extract_frame_at_time(video_path, 0.0)
        
        # Extract last frame (slightly before the end to avoid EOF issues)
        last_time = max(0, duration - 0.1)
        last_frame = extract_frame_at_time(video_path, last_time)
        
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
                    "default": 1, "min": 0, "max": MAX_AUDIO_SLOTS, "step": 1,
                    "tooltip": "Number of audio tracks to mix (0-10). Each gets its own start time and volume."
                }),
                "image_count": ("INT", {
                    "default": 1, "min": 0, "max": MAX_IMAGE_SLOTS, "step": 1,
                    "tooltip": "Number of image overlays (0-5). Each gets its own blend mode, opacity, and position."
                }),
            },
            "optional": {
                # Speed/reverse
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Playback speed. 0.5 = slow motion, 2.0 = double speed."}),
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
                      end_with_audio=True, **kwargs):
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
            filters.append(f"setpts={1/speed}*PTS")
            audio_filters.append(f"atempo={speed}")
        
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
        
        # Fade effects
        if fade_in > 0:
            filters.append(f"fade=t=in:st=0:d={fade_in}")
            audio_filters.append(f"afade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            duration = video['duration'] / speed if speed != 1.0 else video['duration']
            fade_start = max(0, duration - fade_out - 0.05)
            filters.append(f"fade=t=out:st={fade_start}:d={fade_out}")
            audio_filters.append(f"afade=t=out:st={fade_start}:d={fade_out}")
        
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
                'Lighten': 'lighten', 'Difference': 'difference', 'Exclusion': 'exclusion',
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
                    
                    scale_filter = (
                        f"[{input_idx}:v]"
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
                    complex_filter[-1] = f"{scale_filter},colorchannelmixer=aa={opacity},pad={width}:{height}:0:0:color=black@0[{ovr_label}]"
                    complex_filter.append(f"[{current_label}][{ovr_label}]blend=all_mode={ffmpeg_blend}:all_opacity={opacity}[{out_label}]")
                
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
            if filter_complex_str:
                filter_complex_str += ";"
            filter_complex_str += ";".join(audio_mix_parts)
            
            # Determine audio duration mode
            duration_mode = "longest" if end_with_audio else "shortest"
            
            if len(audio_inputs) > 1:
                # Mix multiple audio streams
                filter_complex_str += f";{''.join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration={duration_mode}[aout]"
            elif len(audio_inputs) == 1:
                # Single audio stream
                filter_complex_str = filter_complex_str.replace(audio_inputs[0].replace('[', '').replace(']', ''), 'aout')
        
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
            if audio_filters and video['has_audio']:
                cmd.extend(['-af', ','.join(audio_filters)])
        
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
                "filename_prefix": ("STRING", {"default": "RizzVideo", "tooltip": "Prefix for the output filename. Counter will be appended automatically."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Frames per second for the output video. Ignored if use_source_fps is True."}),
            },
            "optional": {
                "codec": (["libx264", "libx265", "mpeg4", "libvpx-vp9"], {"default": "libx264", "tooltip": "Video codec. libx264 (H.264) is most compatible. libx265 (H.265/HEVC) has better compression. libvpx-vp9 for WebM format."}),
                "quality_crf": ("INT", {"default": 23, "min": 0, "max": 51, "tooltip": "Constant Rate Factor: 0 = lossless, 23 = default, 51 = worst quality. Lower = better quality, larger file."}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "medium", "tooltip": "Encoding speed preset. Slower = better compression but longer encoding time."}),
                "pixel_format": (["yuv420p", "yuv444p"], {"default": "yuv420p", "tooltip": "yuv420p is widely compatible. yuv444p preserves more color detail but less compatible."}),
                "audio_codec": (["aac", "mp3", "opus"], {"default": "aac", "tooltip": "Audio codec. AAC is most compatible with MP4. Opus has best quality at low bitrates."}),
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
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "RizzNodes/Video"
    
    def preview(self, video, autoplay=True, loop=False, max_size=0):
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
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease",
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


# ============================================================================
# Node Mappings
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "RizzLoadVideo": RizzLoadVideo,
    "RizzExtractFrames": RizzExtractFrames,
    "RizzVideoEffects": RizzVideoEffects,
    "RizzSaveVideo": RizzSaveVideo,
    "RizzPreviewVideo": RizzPreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RizzLoadVideo": "Rizz Load Video",
    "RizzExtractFrames": "Rizz Extract First/Last Frames",
    "RizzVideoEffects": "Rizz Video Effects",
    "RizzSaveVideo": "Rizz Save Video",
    "RizzPreviewVideo": "Rizz Preview Video",
}

