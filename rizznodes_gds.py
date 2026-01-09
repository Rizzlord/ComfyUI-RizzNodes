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
            dtype_map = {
                "F16": torch.float16,
                "BF16": torch.bfloat16,
                "F32": torch.float32,
                "I32": torch.int32,
                "I64": torch.int64,
                "I16": torch.int16,
                "I8": torch.int8,
                "U8": torch.uint8,
                "BOOL": torch.bool
            }
            
            if dtype_str not in dtype_map:
                # logging.warning(f"RizzNodes GDS: Unsupported dtype {dtype_str} for tensor {tensor_name}")
                continue
                
            pt_dtype = dtype_map[dtype_str]
            
            # Create empty tensor on GPU
            tensor = torch.empty(size=shape, dtype=pt_dtype, device=device)
            
            # Read directly into tensor buffer
            f_cu.read(tensor, file_offset=data_start + start)
            
            state_dict[tensor_name] = tensor
            
    finally:
        f_cu.close()
        
    return state_dict

def load_torch_file_gds(ckpt, safe_load=False, device=None, return_metadata=False, compat_mode=False):
    """
    Patched version of load_torch_file that attempts to use GDS (GPUDirect Storage)
    via kvikio if available.
    """
    if device is None:
        device = torch.device("cpu")
    
    # Try GDS for safetensors on CUDA
    if GDS_AVAILABLE and device.type == "cuda" and (ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft")):
        try:
            if verbose:
                print(f"### RizzNodes: GDS DMA Transfer Starting for {os.path.basename(ckpt)} ###")

            sd = load_safetensors_gds(ckpt, device, compat_mode=compat_mode)
            
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

    def patch(self, enable, compat_mode, verbose, model=None, clip=None, vae=None, anything=None):
        global _original_load_torch_file
        
        if enable:
            if verbose:
                print(f"### RizzNodes: Patching GDS. Kvikio available: {GDS_AVAILABLE} ###")
            
            def patched_loader(*args, **kwargs):
                if verbose:
                    fname = args[0] if args else "unknown"
                    print(f"### RizzNodes: Intercepted load for {os.path.basename(str(fname))} ###")
                # Pass compat_mode to the loader
                return load_torch_file_gds(*args, **kwargs, compat_mode=compat_mode)
                
            comfy.utils.load_torch_file = patched_loader
            
        else:
            if verbose:
                print("### RizzNodes: Unpatching GDS ###")
            comfy.utils.load_torch_file = _original_load_torch_file

        return (model, clip, vae, anything)