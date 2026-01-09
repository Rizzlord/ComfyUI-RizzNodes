import torch
import os
import logging
import comfy.utils
import comfy.model_management
import folder_paths
import safetensors.torch
import json
import struct

# Try to import GDS libraries
GDS_AVAILABLE = False
try:
    import kvikio
    import kvikio.cufile
    GDS_AVAILABLE = True
except ImportError:
    pass

# Original function backup
_original_load_torch_file = comfy.utils.load_torch_file

def load_safetensors_gds(ckpt, device, compat_mode=False):
    """
    Loads a safetensors file directly to GPU using Kvikio (GDS).
    """
    if not GDS_AVAILABLE:
        raise ImportError("Kvikio not available for GDS load.")

    if compat_mode:
        kvikio.defaults.set_compat_mode(True)

    state_dict = {}
    with open(ckpt, 'rb') as f:
        # Read header length (8 bytes, little-endian unsigned long long)
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {ckpt}")
        
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        
        # Read header
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        
        # Calculate data start position
        data_start = 8 + header_len

    # Open file with kvikio
    f_cu = kvikio.cufile.CuFile(ckpt, "r")
    
    try:
        for tensor_name, info in header.items():
            if tensor_name == "__metadata__":
                continue
                
            # Parse info
            dtype_str = info['dtype']
            shape = info['shape']
            start, end = info['data_offsets']
            
            # Map dtype
            # We use getattr to safely check for newer types
            dtype_map = {
                "F16": torch.float16,
                "BF16": torch.bfloat16,
                "F32": torch.float32,
                "I32": torch.int32,
                "I64": torch.int64,
                "I16": torch.int16,
                "I8": torch.int8,
                "U8": torch.uint8,
                "BOOL": torch.bool,
                "F8_E4M3": getattr(torch, 'float8_e4m3fn', None),
                "F8_E5M2": getattr(torch, 'float8_e5m2', None),
                "F8_E4M3FN": getattr(torch, 'float8_e4m3fn', None),
                "F8_E4M3FNUZ": getattr(torch, 'float8_e4m3fnuz', None),
                "F8_E5M2FNUZ": getattr(torch, 'float8_e5m2fnuz', None),
                "F4_E2M1": getattr(torch, 'float4_e2m1', None),
                "F4_E3M0": getattr(torch, 'float4_e3m0', None),
            }
            
            pt_dtype = dtype_map.get(dtype_str)
            
            # Special Handling for FP8 if PyTorch lacks support
            # ComfyUI often loads FP8 as raw bytes (uint8) if native support is missing, 
            # then casts it later or uses Triton/Bitsandbytes kernels.
            # So if F8 is requested but missing, we try loading as UINT8.
            if pt_dtype is None and dtype_str.startswith("F8"):
                pt_dtype = torch.uint8
            
            if pt_dtype is None:
                # Fallback check for quantized types
                if dtype_str.startswith("Q") or "4" in dtype_str:
                     print(f"### RizzNodes WARNING: Experimental/Quantized dtype '{dtype_str}' detected. Falling back. ###")
                     return _original_load_torch_file(ckpt, device=device)
                
                print(f"### RizzNodes WARNING: Unsupported GDS dtype '{dtype_str}' for tensor '{tensor_name}'. Skipping! ###")
                continue
            
            # Create empty tensor on GPU
            tensor = torch.empty(size=shape, dtype=pt_dtype, device=device)
            
            # Read directly into tensor buffer
            f_cu.read(tensor, file_offset=data_start + start)
            
            state_dict[tensor_name] = tensor

            
            # Create empty tensor on GPU
            tensor = torch.empty(size=shape, dtype=pt_dtype, device=device)
            
            # Read directly into tensor buffer
            f_cu.read(tensor, file_offset=data_start + start)
            
            state_dict[tensor_name] = tensor
            
    finally:
        f_cu.close()
        
    return state_dict

def load_torch_file_gds(ckpt, safe_load=False, device=None, return_metadata=False, compat_mode=False, verbose=False, force_gds=False):
    """
    Patched version of load_torch_file that attempts to use GDS (GPUDirect Storage)
    via kvikio if available.
    """
    if device is None:
        device = torch.device("cpu")
    
    # Determine if we can use GDS
    use_gds = False
    loading_device = device

    # Only applicable for safetensors
    if GDS_AVAILABLE and (ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft")):
        if device.type == "cuda":
            use_gds = True
        elif force_gds and device.type == "cpu" and torch.cuda.is_available():
            # Force GDS means: Load to GPU (GDS), then move to CPU
            use_gds = True
            loading_device = torch.device("cuda")

    if use_gds:
        try:
            if verbose:
                mode_str = "Direct" if loading_device == device else "Staged (GPU->CPU)"
                print(f"### RizzNodes: GDS Start ({mode_str}) for {os.path.basename(ckpt)} -> {loading_device} ###")

            sd = load_safetensors_gds(ckpt, loading_device, compat_mode=compat_mode)
            
            # If we loaded to GPU but wanted CPU, move them now
            if loading_device != device:
                if verbose:
                    print(f"### RizzNodes: Moving {len(sd)} tensors to CPU... ###")
                for k, v in sd.items():
                    sd[k] = v.to(device)
                # Optional: Clear GPU cache after heavy staging
                torch.cuda.empty_cache()

            # Safetensors metadata extraction
            metadata = None
            if return_metadata:
                 with safetensors.safe_open(ckpt, framework="pt", device="cpu") as f:
                     metadata = f.metadata()
            
            if verbose:
                print(f"### RizzNodes: GDS Load SUCCESSFUL ({len(sd)} tensors) ###")
                     
            return (sd, metadata) if return_metadata else sd
            
        except Exception as e:
            logging.error(f"RizzNodes: GDS load failed for {ckpt}, falling back to standard. Error: {e}")
            # Fallthrough to original

    return _original_load_torch_file(ckpt, safe_load, device, return_metadata)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class RizzGDSPatcher:
    DESCRIPTION = "Monkey-patches ComfyUI's file loader to use NVIDIA GPUDirect Storage (via KvikIO) for faster .safetensors loading directly to GPU."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Enable or disable the GDS patch. When enabled, attempts to load safetensors directly to GPU memory."
                }),
                "force_gds": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force GDS even for CPU loads. This loads file -> GPU (via GDS) -> CPU. Can speed up loading if NVMe-GPU link is faster than system IO, but uses VRAM temporarily."
                }),
                "compat_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable KvikIO compatibility mode. Use this if you encounter errors or if GDS is not fully configured on your system (posix fallback)."
                }),
                "verbose": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Print debug messages to the console when files are intercepted and loaded."
                }),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Pass-through for Model (optional, helps define execution order)."}),
                "clip": ("CLIP", {"tooltip": "Pass-through for CLIP (optional)."}),
                "vae": ("VAE", {"tooltip": "Pass-through for VAE (optional)."}),
                "anything": (any_type, {"tooltip": "Pass-through for any other type (optional)."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", any_type)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "anything")
    FUNCTION = "patch"
    CATEGORY = "RizzNodes/System"

    def patch(self, enable, force_gds, compat_mode, verbose, model=None, clip=None, vae=None, anything=None):
        global _original_load_torch_file
        
        if enable:
            if verbose:
                print(f"### RizzNodes: Patching GDS. Kvikio available: {GDS_AVAILABLE} ###")
            
            def patched_loader(*args, **kwargs):
                # Check device if present in args or kwargs to debug
                dev = "unknown"
                if len(args) > 2:
                    dev = args[2]
                elif "device" in kwargs:
                    dev = kwargs["device"]
                
                if verbose:
                    fname = args[0] if args else "unknown"
                    print(f"### RizzNodes: Intercepted load for {os.path.basename(str(fname))} (Requesting: {dev}) ###")
                
                # Pass params to the loader
                return load_torch_file_gds(*args, **kwargs, compat_mode=compat_mode, verbose=verbose, force_gds=force_gds)
                
            comfy.utils.load_torch_file = patched_loader
            
        else:
            if verbose:
                print("### RizzNodes: Unpatching GDS ###")
            comfy.utils.load_torch_file = _original_load_torch_file

        return (model, clip, vae, anything)