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
                "repeat_1": ("BOOLEAN", {"default": False, "label_on": "Repeat (Fill)", "label_off": "No Repeat"}),
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
            inputs["optional"][f"repeat_{i}"] = ("BOOLEAN", {"default": False, "label_on": "Repeat (Fill)", "label_off": "No Repeat"})
            inputs["optional"][f"trim_start_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1})
            inputs["optional"][f"trim_end_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1})

        return inputs

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration")
    FUNCTION = "mix_audio"
    CATEGORY = "RizzNodes/Audio"

    def mix_audio(self, audio_count, audio_1, overall_trim_start=0.0, overall_trim_end=0.0, **kwargs):
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

        # 1. Collect all inputs
        audio_inputs = []
        
        # Audio 1 Processing (Always present)
        vol_1 = kwargs.get("volume_1", 1.0)
        trim_start_1 = kwargs.get("trim_start_1", 0.0)
        trim_end_1 = kwargs.get("trim_end_1", 0.0)
        repeat_1 = kwargs.get("repeat_1", False)
        
        wf_1 = audio_1["waveform"]
        sr_1 = audio_1["sample_rate"]
        
        if trim_start_1 > 0 or trim_end_1 > 0:
            wf_1 = trim_tensor(wf_1, sr_1, trim_start_1, trim_end_1)
            
        audio_data_1 = {**audio_1, "waveform": wf_1}
        audio_inputs.append({
            "data": audio_data_1,
            "mode": True, # Base is always mixed/base
            "volume": vol_1,
            "start_time": 0.0,
            "repeat": repeat_1
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
                repeat = kwargs.get(f"repeat_{i}", False)
                
                # Apply individual trim immediately
                waveform = audio_data["waveform"]
                sr = audio_data["sample_rate"]
                if trim_start > 0 or trim_end > 0:
                     waveform = trim_tensor(waveform, sr, trim_start, trim_end)
                
                audio_data = {**audio_data, "waveform": waveform}
                
                audio_inputs.append({
                    "data": audio_data,
                    "mode": mode, # True=Mix, False=Append
                    "volume": volume,
                    "start_time": start_time,
                    "repeat": repeat
                })
        
        if not audio_inputs:
             return ({"waveform": torch.zeros((1, 2, 44100)), "sample_rate": 44100}, 0.0)

        # 2. Pre-calculate Duration
        # Calculate the expected valid duration of the composition based on NO-REPEAT tracks.
        # Repeating tracks will be expanded to fill this duration.
        
        base_audio = audio_inputs[0]["data"]
        target_sr = base_audio["sample_rate"]
        
        # Simulation loop
        current_sim_len = 0 # In target_sr samples
        
        # Audio 1 Base Duration
        # If Audio 1 is NOT repeating, it provides the baseline length.
        if not audio_inputs[0]["repeat"]:
            wf = audio_inputs[0]["data"]["waveform"]
            l = wf.shape[-1]
            # No resampling needed for base
            current_sim_len = l
            
        for i in range(1, len(audio_inputs)):
            inp = audio_inputs[i]
            is_append = not inp["mode"]
            is_repeat = inp["repeat"]
            
            # Skip duration calculation for repeating tracks (they adapt)
            # Exception: If input is Append mode, Repeat is ignored anyway (can't loop infinite at end)
            # So if Append, we treat as non-repeating for length calc.
            if is_repeat and not is_append:
                continue
                
            original_wf = inp["data"]["waveform"]
            src_sr = inp["data"]["sample_rate"]
            src_len = original_wf.shape[-1]
            
            # Normalize length to target_sr
            if src_sr != target_sr:
                scale = target_sr / src_sr
                len_in_target = int(src_len * scale)
            else:
                len_in_target = src_len
            
            if is_append:
                current_sim_len += len_in_target
            else:
                # Mix Mode
                start_samp = int(inp["start_time"] * target_sr)
                end_samp = start_samp + len_in_target
                current_sim_len = max(current_sim_len, end_samp)
        
        # Fallback: If all tracks are repeating (current_sim_len == 0 but we have inputs),
        # use the length of the first audio (original single cycle) to avoid silence/crash.
        if current_sim_len == 0 and audio_inputs:
             wf = audio_inputs[0]["data"]["waveform"]
             # If base was resampled? No, base is target.
             current_sim_len = wf.shape[-1]

        final_duration_samples = current_sim_len
        if final_duration_samples == 0:
            # Absolute fallback
            final_duration_samples = 1 # Avoid shape 0 errors
        
        # 3. Mixing Loop
        
        # Initialize accumulator with Audio 1
        # If Audio 1 is repeating, we must tile it to final_duration_samples.
        
        curr_wf_1 = audio_inputs[0]["data"]["waveform"].clone()
        if audio_inputs[0]["volume"] != 1.0:
            curr_wf_1 *= audio_inputs[0]["volume"]
            
        if audio_inputs[0]["repeat"]:
            # Tile to final_duration
            target_len = final_duration_samples
            if curr_wf_1.shape[-1] < target_len:
                repeats = (target_len // curr_wf_1.shape[-1]) + 2
                if repeats > 1:
                     # Handle 2D/3D shape for repeat
                     if curr_wf_1.dim() == 2:
                         curr_wf_1 = curr_wf_1.repeat(1, repeats)
                     else:
                         curr_wf_1 = curr_wf_1.repeat(1, 1, repeats)
                curr_wf_1 = curr_wf_1[..., :target_len]
            elif curr_wf_1.shape[-1] > target_len:
                # Case where Repeat=True but target is shorter? Cut it.
                curr_wf_1 = curr_wf_1[..., :target_len]
                
        # If Audio 1 is NOT repeating but is shorter than final_duration (due to other tracks),
        # Pad it? Or just let it be short and 'add' others?
        # Standard mix behavior: start with Audio 1.
        # If we just use Audio 1 as-is, the 'padding' happens during addition of longer tracks.
        current_waveform = curr_wf_1
        current_sr = target_sr
        
        for i in range(1, len(audio_inputs)):
            next_input = audio_inputs[i]
            next_original_waveform = next_input["data"]["waveform"]
            next_sr = next_input["data"]["sample_rate"]
            next_vol = next_input["volume"]
            is_append = not next_input["mode"]
            should_repeat = next_input["repeat"]
            
            # Resample
            if next_sr != current_sr:
                scale_factor = current_sr / next_sr
                new_length = int(next_original_waveform.shape[-1] * scale_factor)
                
                if next_original_waveform.dim() == 2:
                    next_original_waveform = next_original_waveform.unsqueeze(0)
                
                next_wav = torch.nn.functional.interpolate(
                    next_original_waveform, 
                    size=new_length, 
                    mode='linear', 
                    align_corners=False
                )
            else:
                next_wav = next_original_waveform.clone()

            # Handle Repeat (Only in Mix mode)
            if not is_append and should_repeat:
                start_time_sec = next_input["start_time"]
                start_offset = int(start_time_sec * current_sr)
                
                # Available space is defined by final_duration_samples
                # But notice: if we have Append tracks later, final_duration might be longer than current Mix scope?
                # Actually final_duration includes Appends.
                # So if we repeat, we fill the ENTIRE planned duration.
                
                # Wait, if I have Mix(Repeat) then Append. 
                # Does Mix(Repeat) fill the Append part too?
                # Probably yes ("fill available space").
                
                available_space = final_duration_samples - start_offset
                
                if available_space <= 0:
                    next_wav = torch.zeros_like(next_wav)[..., :0]
                else:
                     if next_wav.dim() == 2:
                         next_wav = next_wav.unsqueeze(0)
                        
                     original_len = next_wav.shape[-1]
                     repeats = (available_space // original_len) + 2
                     
                     if repeats > 1:
                         next_wav = next_wav.repeat(1, 1, int(repeats))
                     
                     next_wav = next_wav[..., :available_space]
            
            # Apply volume
            next_wav = next_wav * next_vol
            
            # Apply Start Time (Pre-padding)
            start_time = next_input["start_time"]
            if start_time > 0 and not is_append: # Start time only applies to Mix
                 pad_samples = int(start_time * current_sr)
                 padding = torch.zeros((next_wav.shape[0], next_wav.shape[1], pad_samples), device=next_wav.device)
                 next_wav = torch.cat((padding, next_wav), dim=2)
            
            # Match channels
            if current_waveform.shape[1] == 1 and next_wav.shape[1] > 1:
                current_waveform = current_waveform.repeat(1, next_wav.shape[1], 1)
            elif next_wav.shape[1] == 1 and current_waveform.shape[1] > 1:
                next_wav = next_wav.repeat(1, current_waveform.shape[1], 1)
            
            if is_append:
                current_waveform = torch.cat((current_waveform, next_wav), dim=2)
            else:
                # Mix: Overlap
                # We need to expand current_waveform if next_wav is longer?
                # OR next_wav might be longer than current_waveform IF current_waveform was short (non-repeating Audio 1).
                
                curr_len = current_waveform.shape[-1]
                next_len = next_wav.shape[-1]
                max_len = max(curr_len, next_len)
                
                # Pad current if needed
                if curr_len < max_len:
                    padding = torch.zeros((current_waveform.shape[0], current_waveform.shape[1], max_len - curr_len), device=current_waveform.device)
                    current_waveform = torch.cat((current_waveform, padding), dim=2)
                
                # Pad next if needed
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
