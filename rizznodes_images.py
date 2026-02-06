import os
import torch
import numpy as np
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.utils
import comfy.model_management as mm

class RizzSaveImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (["None", "Flux", "flux2", "qwen", "qwenedit", "sd1.5", "sdxl", "sd3", "anime"],),
                "resize": ("BOOLEAN", {"default": False, "tooltip": "Resize the image to the specified width and height."}),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384, "tooltip": "Target width for resizing."}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384, "tooltip": "Target height for resizing."}),
                "format": (["png", "webp", "jpg", "tga", "bmp"], {"tooltip": "Image format to save as."}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "Compression quality for WebP and JPG."}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Save generation metadata (prompt, workflow) in the image file. Supported formats: PNG, WebP, JPEG."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional upscale model to apply before resizing."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "RizzNodes/Image"

    def save_images(self, images, model, resize, width, height, format, quality, save_metadata=True, upscale_model=None, prompt=None, extra_pnginfo=None):
        return self.save_images_main(images, model, resize, width, height, format, quality, save_metadata, upscale_model, prompt, extra_pnginfo, output_type="output")

    def save_images_main(self, images, model, resize, width, height, format, quality, save_metadata=True, upscale_model=None, prompt=None, extra_pnginfo=None, output_type="output"):
        filename_prefix = model

        if output_type == "output":
            if model == "None":
                output_dir = os.path.join(folder_paths.get_output_directory(), "RizzImage")
                output_subfolder = "RizzImage"
            else:
                output_dir = os.path.join(folder_paths.get_output_directory(), "RizzImage", model)
                output_subfolder = os.path.join("RizzImage", model)
        else: # "temp"
            output_dir = folder_paths.get_temp_directory()
            output_subfolder = ""
            filename_prefix = "RizzPreview" # For temp files

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
        
        results = list()
        
        for image in images:
            # Process image (Resize / Upscale)
            if upscale_model is not None:
                image = self.upscale(image, upscale_model)
                if resize:
                    image = self.resize_image(image, width, height)
            elif resize:
                image = self.resize_image(image, width, height)

            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            metadata = None
            exif_bytes = None
            
            if save_metadata and not get_args_disable_metadata():
                if format == "png":
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                else:
                    # For WebP, JPG, etc. try to save in Exif
                    try:
                        info = {}
                        if prompt is not None:
                            info["prompt"] = prompt
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                info[x] = extra_pnginfo[x]
                        
                        if info:
                            exif = img.getexif()
                            json_str = json.dumps(info)
                            # UserComment (37510) with UNICODE header
                            # header is b"UNICODE\0\0\0" followed by utf-16le encoded string
                            payload = b"UNICODE\0\0\0" + json_str.encode("utf-16le")
                            exif[0x9286] = payload
                            
                            # Also ImageDescription (270) as utf-8 fallback
                            exif[0x010e] = json_str.encode("utf-8")
                            
                            exif_bytes = exif.tobytes()
                            
                    except Exception as e:
                        print(f"Failed to create EXIF data: {e}")

            file = f"{filename}_{counter:05}_.{format}"
            
            # Save logic based on format
            save_path = os.path.join(full_output_folder, file)
            if format == "png":
                 img.save(save_path, pnginfo=metadata, optimize=True)
            elif format == "webp":
                 img.save(save_path, quality=quality, lossless=False, exif=exif_bytes)
            elif format == "jpg":
                 img.save(save_path, quality=quality, optimize=True, exif=exif_bytes)
            elif format == "tga":
                 img.save(save_path)
            elif format == "bmp":
                 img.save(save_path)
            
            # For TGA, create a PNG preview in temp because browsers can't display TGA and we need Alpha support (JPG has no alpha)
            if format == "tga":
                # Create temp preview
                # We need a unique filename for the preview to avoid conflicts if multiple TGAs are saved
                # Use same counter/prefix but in temp dir
                preview_filename = f"RizzPreview_TGA_{filename}_{counter:05}_.png"
                preview_dir = folder_paths.get_temp_directory()
                preview_path = os.path.join(preview_dir, preview_filename)
                
                # Save PNG preview (no metadata needed for preview)
                img.save(preview_path, optimize=True)
                
                results.append({
                    "filename": preview_filename,
                    "subfolder": "", # Empty subfolder implies root of temp dir when type is temp? Or None?
                    # folder_paths.get_save_image_path behavior for temp is a bit specific.
                    # Usually "subfolder" is relative to the base output/temp dir.
                    # If we saved directly to get_temp_directory(), subfolder is empty.
                    "type": "temp"
                })
            else:
                # Resolve subfolder relative to output root (or temp dir if output_type is temp)
                if output_type == "temp":
                    result_subfolder = subfolder if subfolder else ""
                else:
                    if subfolder:
                        result_subfolder = os.path.join(output_subfolder, subfolder)
                    else:
                        result_subfolder = output_subfolder

                results.append({
                    "filename": file,
                    "subfolder": result_subfolder,
                    "type": "temp" if output_type == "temp" else "output"
                })
            
            counter += 1

        return { "ui": { "images": results } }
    
    def resize_image(self, image, width, height):
        # image is [H, W, C] tensor or [1, H, W, C]
        # We need [B, C, H, W] for interpolate usually, or just use PIL
        
        # Ensure BCHW for torch interpolate or HWC for PIL
        # Using PIL for high quality Lanczos
        if len(image.shape) > 3:
            image = image.squeeze(0)
        
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img = img.resize((width, height), resample=Image.LANCZOS)
        
        # Back to tensor
        out = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(out).unsqueeze(0)

    def upscale(self, image, upscale_model):
        # Adaptation of RizzUpscaleImageBatch logic
        # image input here is single [H, W, C] or [1, H, W, C]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        image = image.movedim(-1, 1) # BHWC -> BCHW for model
        
        device = mm.get_torch_device()
        upscale_model.to(device)
        
        tile = 512
        overlap = 32
        
        try:
            # Simple tiled scale
            out = comfy.utils.tiled_scale(image, upscale_model.model, tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=None)
        except Exception as e:
            print(f"Upscale failed: {e}, falling back to original")
            out = image
            
        out = out.movedim(1, -1) # BCHW -> BHWC
        upscale_model.to("cpu")
        return out

class RizzPreviewImage(RizzSaveImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "RizzNodes/Image"

    def save_images(self, images):
        # Persistent preview save (matches Blender preview behavior)
        return self.save_images_main(images, model="Preview", resize=False, width=0, height=0, format="png", quality=90, output_type="output")

    @classmethod
    def VALIDATE_INPUTS(cls, images):
        # Validate inputs to ensure the node doesn't appear empty due to validation issues
        return True

class RizzLoadImage(RizzSaveImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": (["None", "Flux", "flux2", "qwen", "qwenedit", "sd1.5", "sdxl", "sd3", "anime", "Custom"],),
                "custom_path": ("STRING", {"default": "RizzImage"}),
                "image": ([""], {"image_upload": True}), # List instead of STRING for dropdown
                "resize": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "IMAGE/ALPHA")
    FUNCTION = "load_image"
    CATEGORY = "RizzNodes/Image"

    @classmethod
    def VALIDATE_INPUTS(s, image, folder=None, custom_path=None):
        if image in (None, "", "None"):
            return True

        output_root = folder_paths.get_output_directory()

        if folder == "None":
            target_folder = os.path.join(output_root, "RizzImage")
        elif folder == "Custom":
            if custom_path and os.path.isabs(custom_path):
                target_folder = custom_path
            else:
                target_folder = os.path.join(output_root, custom_path or "")
        else:
            target_folder = os.path.join(output_root, "RizzImage", folder or "")

        candidates = [
            os.path.join(target_folder, image),
            os.path.join(folder_paths.get_input_directory(), image),
            os.path.join(output_root, "RizzImage", image),
        ]

        if any(os.path.exists(p) for p in candidates):
            return True

        return f"Invalid image file: {image}"

    def load_image(self, folder, custom_path, image, resize, width, height, upscale_model=None):
        if image == "None" or not image:
             empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
             empty_a = torch.zeros((1, 64, 64, 4), dtype=torch.float32)
             return (empty, empty_a)

        # Determine base folder
        output_root = folder_paths.get_output_directory()
        
        if folder == "None":
            # Search in RizzImage root
            target_folder = os.path.join(output_root, "RizzImage")
        elif folder == "Custom":
            # Use custom path (relative to output or absolute)
            if os.path.isabs(custom_path):
                target_folder = custom_path
            else:
                target_folder = os.path.join(output_root, custom_path)
        else:
            # Specific subfolder (Flux, etc.)
            target_folder = os.path.join(output_root, "RizzImage", folder)

        # Resolve image path
        # If 'image' contains a subfolder separator (uploaded via standard loader sometimes does?)
        # Standard LoadImage usually just gives the filename if it's in input dir, 
        # or relative path if it's in a subdirectory of input.
        # But here we are simulating LoadImage but verifying against our target folder.
        
        # NOTE: standard "image_upload": True widget returns just the filename usually.
        # If we want to support the standard load image behavior, the 'image' arg is the filename.
        # We need to find that file in 'target_folder'.
        
        image_path = os.path.join(target_folder, image)
        
        # Fallback: Check if image path is valid as is (maybe it's absolute from drag drop?)
        # Drag drop usually uploads to 'input' directory by default in standard nodes.
        # But here we want to load from RizzImage. 
        # Crucial realization: "image_upload": True causes the UI to upload dropped files to the SERVER's input directory by default 
        # and return the filename. 
        # If the user drops an image, it goes to `input/`.
        # If the user SELECTS an image from the dropdown, it mimics the file list we provide?
        # A widget cannot cleanly be BOTH a dynamic dropdown from our folder AND an upload widget easily without custom JS.
        # BUT, the user said "drop down images... and ctrl-v as the original load image node".
        # The original LoadImage node uses the `input` directory.
        # Here we want to use `RizzImage` directory.
        
        # If I use `image_upload: True`, ComfyUI treats it as an upload widget.
        # Uploaded files go to `input`. 
        # Existing files in `input` are listed.
        # To list files from `RizzImage`, we need our custom JS to populate the widget.
        # If we successfully populate it with files from `RizzImage`, selecting one sends that filename.
        # BUT if user pastes/drops, Comfy uploads to `input` and selects it.
        # So `image_path` checks:
        # 1. target_folder (RizzImage/...)
        # 2. input folder (for uploads)
        
        if not os.path.exists(image_path):
            # Check input folder for uploads
            input_dir = folder_paths.get_input_directory()
            input_path = os.path.join(input_dir, image)
            if os.path.exists(input_path):
                image_path = input_path
            else:
                 # Last ditch: try finding it in RizzImage root if slightly misplaced
                 root_path = os.path.join(output_root, "RizzImage", image)
                 if os.path.exists(root_path):
                     image_path = root_path
                 else:
                     raise FileNotFoundError(f"Image not found: {image}")

        img = Image.open(image_path)
        
        # Helper for resize
        def process_resize(image_pil, w, h):
            return image_pil.resize((w, h), resample=Image.LANCZOS)
            
        # Convert to tensor first for upscale helper
        i = np.array(img).astype(np.float32) / 255.0
        # Check channels
        if i.ndim == 2: # L
             i = i[..., None]
        
        # Upscale expects tensor input [1, H, W, C]? No, my upscale helper expects [B, H, W, C]
        img_tensor = torch.from_numpy(i).unsqueeze(0)
        
        # If RGBA, we need to handle alpha carefully or drop it for Upscale Model usually
        # Upscale models work on RGB usually.
        # RizzSaveImage logic uses self.upscale(image, model)
        
        if upscale_model is not None:
             # Upscale model usually RGB.
             # If input has alpha, maybe separate, upscale both? 
             # For simplicity, let's upscale RGB. Alpha might result in weirdness if upscaled by model.
             # User requested "resize with lanczos or upscale model".
             
             # Convert to RGB for upscale
             img_rgb_t = img_tensor[..., :3]
             if img_tensor.shape[-1] == 4:
                  img_alpha_t = img_tensor[..., 3:]
             else:
                  img_alpha_t = None
             
             img_rgb_t = self.upscale(img_rgb_t, upscale_model)
             
             # Alpha upscale - just bicubic/lanczos to match new size?
             if img_alpha_t is not None:
                  # Resize alpha to match new RGB size
                  t_h, t_w = img_rgb_t.shape[1], img_rgb_t.shape[2] # BHWC
                  # Need BCHW for interpolate
                  alpha_chw = img_alpha_t.movedim(-1, 1)
                  alpha_up = torch.nn.functional.interpolate(alpha_chw, size=(t_h, t_w), mode="bilinear")
                  img_alpha_t = alpha_up.movedim(1, -1)
                  
                  # Recombine
                  img_tensor = torch.cat((img_rgb_t, img_alpha_t), dim=-1)
             else:
                  img_tensor = img_rgb_t
             
             if resize:
                 # Post-upscale resize?
                 img_tensor = self.resize_image(img_tensor, width, height)
                 
        elif resize:
             img_tensor = self.resize_image(img_tensor, width, height)
             
        # Separate RGB/RGBA for output
        # img_tensor is [1, H, W, C]
        if img_tensor.shape[-1] == 4:
             tensor_rgba = img_tensor
             tensor_rgb = img_tensor[..., :3]
        else:
             tensor_rgb = img_tensor
             # Create solid alpha
             alpha = torch.ones_like(tensor_rgb[..., 0:1])
             tensor_rgba = torch.cat((tensor_rgb, alpha), dim=-1)
             
        image_type = "output" if image_path.startswith(folder_paths.get_output_directory()) else "input"
        if image_type == "output":
            subfolder = os.path.dirname(os.path.relpath(image_path, folder_paths.get_output_directory()))
        else:
            subfolder = os.path.dirname(os.path.relpath(image_path, folder_paths.get_input_directory()))

        return {
            "ui": {
                "images": [
                    {
                        "filename": os.path.basename(image_path),
                        "subfolder": subfolder,
                        "type": image_type
                    }
                ]
            },
            "result": (tensor_rgb, tensor_rgba)
        }

    @classmethod
    def IS_CHANGED(s, folder, custom_path, image, resize, width, height, upscale_model=None):
        import hashlib
        m = hashlib.sha256()
        m.update(folder.encode())
        m.update(custom_path.encode())
        m.update(image.encode())
        return m.digest().hex()

# Shim for args
import hashlib
args_disable_metadata = False

# Define globally for all classes in this module
def get_args_disable_metadata():
    return args_disable_metadata

# ============================================================================
# Blend Mode Functions (PyTorch)
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
    mask = (base < 0.5).float()
    result = 2 * base * overlay * mask + (1 - 2 * (1 - base) * (1 - overlay)) * (1 - mask)
    return base * (1 - opacity) + result * opacity

def blend_soft_light(base, overlay, opacity):
    # result = (1 - 2*overlay)*base*base + 2*base*overlay (simplified approx often used)
    # OR standard Photoshop formula:
    # if overlay < 0.5: base - (1 - 2 * overlay) * base * (1 - base)
    # else: base + (2 * overlay - 1) * (sqrt(base) - base)
    
    mask = (overlay < 0.5).float()
    term1 = base - (1 - 2 * overlay) * base * (1 - base)
    term2 = base + (2 * overlay - 1) * (torch.sqrt(base + 1e-6) - base)
    result = term1 * mask + term2 * (1 - mask)
    return base * (1 - opacity) + result * opacity

def blend_hard_light(base, overlay, opacity):
    # Swap base and overlay in Overlay formula
    # if overlay < 0.5: 2 * base * overlay
    # else: 1 - 2 * (1 - base) * (1 - overlay)
    mask = (overlay < 0.5).float()
    result = 2 * base * overlay * mask + (1 - 2 * (1 - base) * (1 - overlay)) * (1 - mask)
    return base * (1 - opacity) + result * opacity

def blend_color_dodge(base, overlay, opacity):
    # base / (1 - overlay)
    result = base / (1 - overlay + 1e-6)
    result = torch.clamp(result, 0, 1)
    return base * (1 - opacity) + result * opacity

def blend_color_burn(base, overlay, opacity):
    # 1 - (1 - base) / overlay
    result = 1 - (1 - base) / (overlay + 1e-6)
    result = torch.clamp(result, 0, 1)
    return base * (1 - opacity) + result * opacity

def blend_darken(base, overlay, opacity):
    result = torch.min(base, overlay)
    return base * (1 - opacity) + result * opacity

def blend_lighten(base, overlay, opacity):
    result = torch.max(base, overlay)
    return base * (1 - opacity) + result * opacity

# Helper for dispatch
BLEND_MODES_FUNC = {
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

MAX_IMAGE_SLOTS = 5

class RizzImageEffects:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "base_image": ("IMAGE",),
                "image_count": ("INT", {
                    "default": 0, "min": 0, "max": MAX_IMAGE_SLOTS, "step": 1,
                    "tooltip": "Number of image overlays (0-5). Each gets its own blend mode, opacity, and position."
                }),
            },
            "optional": {}
        }
        
        blend_modes = list(BLEND_MODES_FUNC.keys())
        position_modes = ["Stretched", "Tiled", "Center", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        
        # Generate dynamic inputs
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

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effects"
    CATEGORY = "RizzNodes/Image"

    def apply_effects(self, base_image, image_count=0, **kwargs):
        # Base image: [B, H, W, C]
        temp_image = base_image.clone()
        
        batch_size, h, w, c = temp_image.shape
        
        # We process each overlay
        for i in range(1, min(image_count, MAX_IMAGE_SLOTS) + 1):
            overlay_img = kwargs.get(f"image_{i}")
            if overlay_img is None:
                continue
                
            blend_mode_name = kwargs.get(f"image_{i}_blend", "Normal")
            opacity = kwargs.get(f"image_{i}_opacity", 1.0)
            position = kwargs.get(f"image_{i}_position", "Stretched")
            tile_scale = kwargs.get(f"image_{i}_tile_scale", 1.0)
            
            # Prepare overlay
            # overlay_img is [B_o, H_o, W_o, C_o]
            # If B_o != B, we might need to handle broadcasting or just use first frame?
            # Comfy usually broadcasts if one is 1. If both > 1 and mismatch, error or clip?
            # We'll rely on simple iter logic or broadcasting.
            
            # 1. Resize/Transform overlay to match base (h, w)
            
            # Helper to resize tensor [B, H, W, C] -> [B, h, w, C]
            def resize_to(img_tensor, new_h, new_w):
                # Permute to [B, C, H, W] for grid_sample/interpolate
                img_p = img_tensor.permute(0, 3, 1, 2)
                img_r = torch.nn.functional.interpolate(img_p, size=(new_h, new_w), mode='bilinear', align_corners=False)
                return img_r.permute(0, 2, 3, 1) # Back to [B, H, W, C]
            
            processed_overlay = None
            
            ov_h, ov_w = overlay_img.shape[1], overlay_img.shape[2]
            
            if position == "Stretched":
                processed_overlay = resize_to(overlay_img, h, w)
                
            elif position == "Tiled":
                # Calculate new size based on scale
                # scale 1.0 means original size? User tooltip says "Smaller = more tiles".
                # But logic says "Scale factor for tiled image".
                # If scale=1.0, utilize original dimensions?
                # Let's interpret scale as: target_size = original_size * scale
                
                target_h = max(1, int(ov_h * tile_scale))
                target_w = max(1, int(ov_w * tile_scale))
                
                # Resize overlay to target size
                resized_ov = resize_to(overlay_img, target_h, target_w)
                
                # Tile it to fill (h, w)
                # repeat counts
                import math
                rep_y = math.ceil(h / target_h)
                rep_x = math.ceil(w / target_w)
                
                # Repeat
                tiled = resized_ov.repeat(1, rep_y, rep_x, 1)
                
                # Crop to exact fit
                processed_overlay = tiled[:, :h, :w, :]
                
            else: # Center, Corners
                # Keep aspect ratio / original size, but place it
                # We need a canvas of size (h, w) separate for the overlay?
                # Or we can use padding logic.
                
                # First, ensure we don't resize the overlay unless we want to?
                # "Position/sizing mode" implies moving it.
                # Video node: scale='min(W,iw)':'min(H,ih)' -> fit inside but keep aspect?
                # Or just keep original size if smaller?
                # Let's keep original size (clipped if larger).
                
                # Create empty canvas (transparent black)
                # If overlay has alpha... our tensors are usually RGB (3 channels) or RGBA (4).
                # Comfy generally uses RGB images unless loaded with mask.
                # If overlay_img has 3 channels, we assume opaque?
                # If we put a small image on top, the rest is transparent (for blending).
                
                # We need an alpha channel for the overlay layer to composite correctly over base.
                
                # Case 1: result buffer [B, H, W, C]. Initialize with zeros (transparent).
                # But wait, blend modes work on pixels.
                # If we have "Center", we want the overlay pixels in center, and 0 contribution elsewhere?
                # Yes, effectively opacity 0 elsewhere.
                
                # Let's construct a full-size overlay layer [B, H, W, C]
                # Filled with... what?
                # For "Normal" blend, transparent pixels don't affect base.
                # For "Multiply", transparent pixels should be... 1.0 (white)?
                # Comfy blend nodes usually use masks.
                # Here we are simulating layers.
                
                # We need a mask channel for the overlay.
                if overlay_img.shape[-1] == 4:
                    ov_rgb = overlay_img[..., :3]
                    ov_a = overlay_img[..., 3:4]
                else:
                    ov_rgb = overlay_img
                    ov_a = torch.ones((overlay_img.shape[0], ov_h, ov_w, 1), device=overlay_img.device)
                
                # Create output buffers
                canvas_rgb = torch.zeros((batch_size, h, w, 3), device=temp_image.device)
                canvas_a = torch.zeros((batch_size, h, w, 1), device=temp_image.device)
                
                # Calculate positions
                pad_top, pad_left = 0, 0
                
                eff_h = min(h, ov_h)
                eff_w = min(w, ov_w)
                
                if position == "Center":
                    pad_top = (h - eff_h) // 2
                    pad_left = (w - eff_w) // 2
                elif position == "Top-Left":
                    pad_top = 0
                    pad_left = 0
                elif position == "Top-Right":
                    pad_top = 0
                    pad_left = w - eff_w
                elif position == "Bottom-Left":
                    pad_top = h - eff_h
                    pad_left = 0
                elif position == "Bottom-Right":
                    pad_top = h - eff_h
                    pad_left = w - eff_w
                
                # Place overlay on canvas
                # We take the crop of overlay that fits
                # Src slice: 0 to eff_h, 0 to eff_w
                # Dst slice: pad_top to pad_top+eff_h, ...
                
                # Handle batch mismatch simply by repeating if needed or modulo
                for b in range(batch_size):
                    ov_idx = b % overlay_img.shape[0]
                    canvas_rgb[b, pad_top:pad_top+eff_h, pad_left:pad_left+eff_w, :] = ov_rgb[ov_idx, :eff_h, :eff_w, :]
                    canvas_a[b, pad_top:pad_top+eff_h, pad_left:pad_left+eff_w, :] = ov_a[ov_idx, :eff_h, :eff_w, :]
                
                processed_overlay = torch.cat((canvas_rgb, canvas_a), dim=-1)

            # Ensure broadcasting if batch sizes differ for Stretched/Tiled cases
            if processed_overlay.shape[0] != batch_size:
                 # Simple repeat if 1 -> N
                 if processed_overlay.shape[0] == 1:
                     processed_overlay = processed_overlay.repeat(batch_size, 1, 1, 1)
                 else:
                     # If mismatch otherwise, maybe slice?
                     # Just force match
                     pass
            
            # Perform Blend
            # Base is RGB or RGBA?
            if temp_image.shape[-1] == 4:
                base_rgb = temp_image[..., :3]
            else:
                base_rgb = temp_image
                
            if processed_overlay.shape[-1] == 4:
                ov_rgb = processed_overlay[..., :3]
                ov_a = processed_overlay[..., 3:4] # Mask from positioning/alpha
            else:
                ov_rgb = processed_overlay
                ov_a = torch.ones((batch_size, h, w, 1), device=temp_image.device)
            
            # 1. Apply opacity to the overlay alpha
            final_ov_a = ov_a * opacity
            
            # 2. Blend
            # Standard composition: Result = Blend(Base, Overlay) * OvAlpha + Base * (1 - OvAlpha)
            # The blend function gives the color where overlay exists.
            
            func = BLEND_MODES_FUNC.get(blend_mode_name, blend_normal)
            blended_rgb = func(base_rgb, ov_rgb, 0) # Opacity handled by alpha mix below
             # Passing 0 as opacity to the blend func because we use alpha compositing for the final mix.
             # Wait, standard blend functions usually output the "blended result".
             # Opacity param in my functions is `base * (1-op) + res * op`.
             # If I pass 0, I get `base`. That's wrong. I want `res`.
             # If I pass 1, I get `res`.
             # So blend_rgb = func(base, overlay, 1.0)
            
            blended_rgb = func(base_rgb, ov_rgb, 1.0)
            
            # Compose
            final_rgb = blended_rgb * final_ov_a + base_rgb * (1 - final_ov_a)
            
            # Update base image
            if temp_image.shape[-1] == 4:
                # Update alpha? Usually keep base alpha or union? 
                # Let's say we keep base alpha logic or just 1.0 if not.
                temp_image = torch.cat((final_rgb, temp_image[..., 3:4]), dim=-1)
            else:
                temp_image = final_rgb

        return (temp_image,)
