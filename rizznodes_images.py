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
                "resize": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "format": (["png", "webp", "jpg", "tga", "bmp"],),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
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
            else:
                output_dir = os.path.join(folder_paths.get_output_directory(), "RizzImage", model)
        else: # "temp"
            output_dir = folder_paths.get_temp_directory()
            filename_prefix = "RizzPreview" # For temp files

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
        
        results = list()
        
        for image in images:
            # Process image (Resize / Upscale)
            if upscale_model is not None:
                image = self.upscale(image, upscale_model)
                # If resize is ALSO checked, maybe we should resize AFTER upscale?
                # User logic: "resize with lanczos or upscale model".
                # Implies one or the other. But if upscale is connected, it probably takes precedence for "upscaling".
                # If user connects upscale model AND sets resize=True with specific W/H, 
                # strictly speaking we should probably downscale/resize to that target after upscaling?
                # For now, let's assume upscale model handles the resolution increase.
                # If the user WANTS to force a size, they can check resize.
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
                        import piexif
                        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
                        
                        info = {}
                        if prompt is not None:
                            info["prompt"] = prompt
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                info[x] = extra_pnginfo[x]
                        
                        if info:
                            # UserComment tag ID is 37510
                            user_comment = json.dumps(info)
                            exif_dict["Exif"][37510] = user_comment.encode("utf-8")
                            exif_bytes = piexif.dump(exif_dict)
                            
                    except ImportError:
                        # Fallback if piexif not installed
                        pass
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
            
            results.append({
                "filename": file,
                "subfolder": "RizzImage" if model == "None" else os.path.join("RizzImage", model),
                "type": "output"
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
        # Simple temp save for preview
        return self.save_images_main(images, model="Preview", resize=False, width=0, height=0, format="png", quality=90, output_type="temp")

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
                "image": (["None"], {"image_upload": True}), # List instead of STRING for dropdown
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
             
        return (tensor_rgb, tensor_rgba), {
            "ui": {
                "images": [
                    {
                        "filename": os.path.basename(image_path),
                        "subfolder": os.path.dirname(os.path.relpath(image_path, folder_paths.get_output_directory())),
                        "type": "output" if "output" in image_path else "input"
                    }
                ]
            }
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
