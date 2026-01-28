
import sys
import os
import torch
import numpy as np

# Add paths
sys.path.append("/Apps/ComfyUI")
sys.path.append("/Apps/ComfyUI/custom_nodes/ComfyUI-RizzNodes")
from rizznodes_videosuit import RizzSeparateVideoAudio, RizzExtractAllFrames, get_video_info

def create_dummy_video(filename="dummy_test.mp4", duration=1.0, fps=24):
    import subprocess
    # Create video with audio
    cmd = [
        'ffmpeg', '-y', '-f', 'lavfi', '-i', f'testsrc=duration={duration}:size=320x240:rate={fps}',
        '-f', 'lavfi', '-i', f'sine=frequency=1000:duration={duration}',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-shortest',
        filename
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return os.path.abspath(filename)

def verify_nodes():
    print("Creating dummy video with audio...")
    video_path = create_dummy_video()
    video_info = get_video_info(video_path)
    
    print(f"Video created: {video_path}, Has Audio: {video_info['has_audio']}")
    
    # Test Separation
    print("\n--- Testing RizzSeparateVideoAudio ---")
    sep_node = RizzSeparateVideoAudio()
    video_out, audio_out = sep_node.separate(video_info)
    
    print(f"Video Output Path: {video_out['path']}")
    print(f"Video Output Has Audio: {video_out['has_audio']}")
    if not video_out['has_audio']:
        print("SUCCESS: Video output is muted.")
    else:
        print("FAILURE: Video output still has audio.")
    
    waveform = audio_out['waveform']
    sr = audio_out['sample_rate']
    print(f"Audio Output Waveform Shape: {waveform.shape}")
    print(f"Audio Output Sample Rate: {sr}")
    
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        print("SUCCESS: Audio shape is [1, C, T] compliant.")
    else:
        print("FAILURE: Audio shape mismatch.")

    # Test Extraction
    print("\n--- Testing RizzExtractAllFrames ---")
    ext_node = RizzExtractAllFrames()
    
    # Extract ALL
    images, = ext_node.extract_all(video_info, limit_frames=0)
    print(f"Extracted Frames Shape: {images.shape}")
    
    expected_frames = int(video_info['duration'] * video_info['fps'])
    if abs(images.shape[0] - expected_frames) <= 2:
        print("SUCCESS: Extracted frame count matches expected.")
    else:
        print(f"WARNING: Frame count mismatch. Expected ~{expected_frames}, got {images.shape[0]}")

    # Test Limit
    limit = 5
    print(f"\n--- Testing RizzExtractAllFrames (Limit {limit}) ---")
    images_lim, = ext_node.extract_all(video_info, limit_frames=limit)
    print(f"Extracted Limited Frames Shape: {images_lim.shape}")
    
    if images_lim.shape[0] == limit:
        print("SUCCESS: Frame limit respected.")
    else:
        print("FAILURE: Frame limit ignored.")

if __name__ == "__main__":
    try:
        verify_nodes()
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
