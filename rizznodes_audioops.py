import torch

# Maximum number of audio inputs to expose
MAX_AUDIO_INPUTS = 50

class RizzAudioMixer:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "audio_count": ("INT", {"default": 2, "min": 1, "max": MAX_AUDIO_INPUTS, "step": 1}),
                "audio_1": ("AUDIO",),
            },
            "optional": {
                "overall_trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "overall_trim_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                # Audio 1 optional settings (Mode 1 is not needed as it's the base)
                "volume_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "trim_start_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "trim_end_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
            }
        }

        # Dynamically add audio inputs starting from 2
        for i in range(2, MAX_AUDIO_INPUTS + 1):
            inputs["optional"][f"audio_{i}"] = ("AUDIO",)
            inputs["optional"][f"mode_{i}"] = ("BOOLEAN", {"default": True, "label_on": "Mix (Overlap)", "label_off": "Append (Sequence)"})
            inputs["optional"][f"start_time_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1})
            inputs["optional"][f"volume_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})
            inputs["optional"][f"trim_start_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1})
            inputs["optional"][f"trim_end_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1})

        return inputs

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration")
    FUNCTION = "mix_audio"
    CATEGORY = "RizzNodes/Audio"

    def mix_audio(self, audio_count, audio_1, overall_trim_start=0.0, overall_trim_end=0.0, **kwargs):
        # Initialize accumulator
        
        # Helper to trim audio tensor [batch, channels, samples]
        def trim_tensor(audio_tensor, sample_rate, start_sec, end_sec):
            total_samples = audio_tensor.shape[-1]
            start_sample = int(start_sec * sample_rate)
            
            if start_sample >= total_samples:
                return torch.zeros_like(audio_tensor)[:, :, :0]
                
            end_sample = total_samples
            if end_sec > 0:
                end_cut = int(end_sec * sample_rate)
                end_sample = max(start_sample, total_samples - end_cut)
            
            return audio_tensor[..., start_sample:end_sample]

        # Process audio_1 (Always present)
        audio_inputs = []
        
        # Audio 1 Processing
        vol_1 = kwargs.get("volume_1", 1.0)
        trim_start_1 = kwargs.get("trim_start_1", 0.0)
        trim_end_1 = kwargs.get("trim_end_1", 0.0)
        
        wf_1 = audio_1["waveform"]
        sr_1 = audio_1["sample_rate"]
        
        if trim_start_1 > 0 or trim_end_1 > 0:
            wf_1 = trim_tensor(wf_1, sr_1, trim_start_1, trim_end_1)
            
        audio_data_1 = {**audio_1, "waveform": wf_1}
        audio_inputs.append({
            "data": audio_data_1,
            "mode": True, # Base is always mixed/base
            "volume": vol_1,
            "start_time": 0.0 # Audio 1 always starts at 0
        })

        # Process remaining inputs
        for i in range(2, audio_count + 1):
            audio_data = kwargs.get(f"audio_{i}")
            if audio_data is not None:
                mode = kwargs.get(f"mode_{i}", True)
                volume = kwargs.get(f"volume_{i}", 1.0)
                trim_start = kwargs.get(f"trim_start_{i}", 0.0)
                trim_end = kwargs.get(f"trim_end_{i}", 0.0)
                start_time = kwargs.get(f"start_time_{i}", 0.0)
                
                # Apply individual trim immediately
                waveform = audio_data["waveform"]
                sr = audio_data["sample_rate"]
                if trim_start > 0 or trim_end > 0:
                     waveform = trim_tensor(waveform, sr, trim_start, trim_end)
                
                # Update audio_data with trimmed waveform
                audio_data = {**audio_data, "waveform": waveform}
                
                audio_inputs.append({
                    "data": audio_data,
                    "mode": mode, # True=Mix, False=Append
                    "volume": volume,
                    "start_time": start_time
                })
        
        if not audio_inputs:
             return ({"waveform": torch.zeros((1, 2, 44100)), "sample_rate": 44100}, 0.0)

        # Base properties from first audio
        base_audio = audio_inputs[0]["data"]
        
        # Helper to normalize waveform to [batch, channels, samples]
        def get_waveform_tensor(audio_dict):
            wf = audio_dict["waveform"]
            # Handle standard ComfyUI format: [batch, channels, len] or [batch, len, channels] ?
            # Most nodes output [batch, channels, length].
            # rizznodes_videosuit handles:
            # if waveform.ndim == 3: waveform = waveform[0] (Remove batch)
            # if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]: waveform = waveform.T
            
            # We want to work with [channels, length] for mixing single batch items, 
            # OR [batch, channels, length] if we want to be correct.
            # Let's standardise to [batch, channels, length].
            
            if wf.dim() == 2:
                # Assuming [channels, length] or [length, samples]?
                # Usually [channels, length] coming from torchaudio.load
                pass 
                
            return wf.clone() 

        target_sr = base_audio["sample_rate"]
        
        # Start with the first audio
        # We process the list sequentially
        
        # Accumulator:
        current_waveform = audio_inputs[0]["data"]["waveform"].clone()
        if audio_inputs[0]["volume"] != 1.0:
            current_waveform *= audio_inputs[0]["volume"]
            
        current_sr = audio_inputs[0]["data"]["sample_rate"]

        for i in range(1, len(audio_inputs)):
            next_input = audio_inputs[i]
            next_original_waveform = next_input["data"]["waveform"]
            next_sr = next_input["data"]["sample_rate"]
            next_vol = next_input["volume"]
            is_append = not next_input["mode"]
            
            # Check dimensions
            # Ensure shape matches [batch, channels, length]
            # If current is 1 channel and next is 2, or vice versa, we might need adjustments.
            # For simplicity, if channel mismatch, maybe duplicate mono to stereo?
            
            # Resampling check and handling
            if next_sr != current_sr:
                # Calculate new length
                # samples_target = samples_source * (target_sr / source_sr)
                scale_factor = current_sr / next_sr
                new_length = int(next_original_waveform.shape[-1] * scale_factor)
                
                # Use interpolate (needs 3D input [batch, channels, time])
                # next_original_waveform is [batch, channels, time]
                if next_original_waveform.dim() == 2:
                    next_original_waveform = next_original_waveform.unsqueeze(0)
                
                next_wav = torch.nn.functional.interpolate(
                    next_original_waveform, 
                    size=new_length, 
                    mode='linear', 
                    align_corners=False
                )
                # Next tensor is now at current_sr
            else:
                next_wav = next_original_waveform.clone()

            # Apply volume
            next_wav = next_wav * next_vol
            
            # Apply Start Time (Pre-padding)
            # We must pad with zeros equivalent to start_time seconds
            # Note: This is applied AFTER resampling to current_sr
            start_time = next_input["start_time"]
            if start_time > 0:
                 pad_samples = int(start_time * current_sr)
                 padding = torch.zeros((next_wav.shape[0], next_wav.shape[1], pad_samples), device=next_wav.device)
                 next_wav = torch.cat((padding, next_wav), dim=2)
            
            # Match channels (Naive approach: Expand 1->2 if needed)
            if current_waveform.shape[1] == 1 and next_wav.shape[1] > 1:
                current_waveform = current_waveform.repeat(1, next_wav.shape[1], 1)
            elif next_wav.shape[1] == 1 and current_waveform.shape[1] > 1:
                next_wav = next_wav.repeat(1, current_waveform.shape[1], 1)
            
            if is_append:
                # Append: Concatenate along last dim (time)
                # Assumes dim 2 is time: [batch, channels, time]
                current_waveform = torch.cat((current_waveform, next_wav), dim=2)
            else:
                # Mix: Overlap
                # Pad the shorter one to match longer one
                curr_len = current_waveform.shape[2]
                next_len = next_wav.shape[2]
                
                max_len = max(curr_len, next_len)
                
                # Check formatting. torch.cat needs same dimensions.
                # Padding:
                if curr_len < max_len:
                    padding = torch.zeros((current_waveform.shape[0], current_waveform.shape[1], max_len - curr_len), device=current_waveform.device)
                    current_waveform = torch.cat((current_waveform, padding), dim=2)
                
                if next_len < max_len:
                    padding = torch.zeros((next_wav.shape[0], next_wav.shape[1], max_len - next_len), device=next_wav.device)
                    next_wav = torch.cat((next_wav, padding), dim=2)
                    
                current_waveform = current_waveform + next_wav
        
        # Apply overall trim
        if overall_trim_start > 0 or overall_trim_end > 0:
            current_waveform = trim_tensor(current_waveform, current_sr, overall_trim_start, overall_trim_end)

        duration = current_waveform.shape[-1] / current_sr
        return ({"waveform": current_waveform, "sample_rate": current_sr}, float(duration))

NODE_CLASS_MAPPINGS = {
    "RizzAudioMixer": RizzAudioMixer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RizzAudioMixer": "Audio Mixer (Rizz)"
}
