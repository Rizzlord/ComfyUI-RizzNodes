import os
import torch
from PIL import Image, ImageFilter
import numpy as np
import pymeshlab as ml
import trimesh
from trimesh.proximity import ProximityQuery
import logging
import re
import gc
import comfy.model_management as mm
import comfy.utils
import folder_paths
import scipy.ndimage

logging.getLogger('trimesh').setLevel(logging.ERROR)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

_NODE_STATE = {}

def rgb_to_hsv(arr):
    arr = np.asarray(arr, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    v = maxc
    s = (maxc - minc) / (maxc + 1e-10)
    s[maxc == 0] = 0
    rc = (maxc - r) / (maxc - minc + 1e-10)
    gc = (maxc - g) / (maxc - minc + 1e-10)
    bc = (maxc - b) / (maxc - minc + 1e-10)
    h = np.zeros_like(maxc)
    mask = maxc == r
    h[mask] = np.mod(0.0 + (gc - bc)[mask], 6.0)
    mask = maxc == g
    h[mask] = 2.0 + (bc - rc)[mask]
    mask = maxc == b
    h[mask] = 4.0 + (rc - gc)[mask]
    h = h / 6.0
    h = np.mod(h, 1.0)
    h[np.isnan(h)] = 0.0
    s[np.isnan(s)] = 0.0
    return np.stack([h, s, v], axis=-1)

def hsv_to_rgb(arr):
    arr = np.asarray(arr, dtype=np.float32)
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    h = h * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - (s * f))
    t = v * (1.0 - (s * (1.0 - f)))
    i = np.mod(i, 6)
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    mask = i == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = i == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = i == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = i == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = i == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = i == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
    return np.stack([r, g, b], axis=-1)
class RizzLoadLatestImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "filename_base": ("STRING", {"default": "Image"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "latest_filename")
    FUNCTION = "load_latest_image"
    CATEGORY = "RizzNodes/Image"

    def load_latest_image(self, directory, filename_base):
        if not os.path.isabs(directory):
            search_path = os.path.join(folder_paths.get_output_directory(), directory)
        else:
            search_path = directory

        if not os.path.isdir(search_path):
            return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)), "Directory Not Found")

        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff')
        pattern = re.compile(rf"{re.escape(filename_base)}_(\d+).*?(?i:({'|'.join(ext.lstrip('.') for ext in image_extensions)}))$")
        
        highest_num = -1
        latest_file = None

        for filename in os.listdir(search_path):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num
                    latest_file = filename

        if latest_file:
            file_path = os.path.join(search_path, latest_file)
            try:
                img = Image.open(file_path)
                img = img.convert("RGBA")
                image_np = np.array(img, dtype=np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                
                mask = image_tensor[:, :, :, 3]
                image_rgb = image_tensor[:, :, :, :3]

                return (image_rgb, mask, latest_file)
            except Exception as e:
                return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)), f"Error loading: {e}")
        else:
            return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)), "No matching file found")


class RizzLoadLatestMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "filename_base": ("STRING", {"default": "Unrefined"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("MESH", "latest_filename")
    FUNCTION = "load_latest_mesh"
    CATEGORY = "RizzNodes/Model"

    def load_latest_mesh(self, directory, filename_base):
        if not os.path.isabs(directory):
            search_path = os.path.join(folder_paths.get_output_directory(), directory)
        else:
            search_path = directory

        if not os.path.isdir(search_path):
            return (trimesh.Trimesh(), "Directory Not Found")

        pattern = re.compile(rf"{re.escape(filename_base)}_(\d+).*?\.glb$")

        highest_num = -1
        latest_file = None

        for filename in os.listdir(search_path):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num
                    latest_file = filename

        if latest_file:
            file_path = os.path.join(search_path, latest_file)
            try:
                mesh = trimesh.load(file_path, force='mesh')
                return (mesh, latest_file)
            except Exception as e:
                return (trimesh.Trimesh(), f"Error loading: {e}")
        else:
            return (trimesh.Trimesh(), "No matching file found")

class RizzBatchImageLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "input/images"}),
            },
            "optional": {
                "reset_index": ("BOOLEAN", {"default": False, "button": True, "label": "Reset Index"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STRING", "INT", "INT",)
    RETURN_NAMES = ("CURRENT_IMAGE", "CURRENT_MASK", "IMAGE_BATCH", "MASK_BATCH", "CURRENT_FILENAME", "CURRENT_INDEX", "TOTAL_IMAGES",)
    FUNCTION = "load_next_image"
    CATEGORY = "RizzNodes/Image"

    def load_next_image(self, directory, reset_index=False):
        instance_id = id(self)
        if instance_id not in _NODE_STATE:
            _NODE_STATE[instance_id] = {
                "current_index": -1,
                "image_files": [],
                "last_directory": None,
                "current_image_data": None
            }
        state = _NODE_STATE[instance_id]

        if reset_index or directory != state["last_directory"]:
            state["current_index"] = -1
            state["last_directory"] = directory
            state["image_files"] = []

        base_dir = folder_paths.get_input_directory() if not os.path.isabs(directory) else directory
        
        if not state["image_files"]:
            if not os.path.isdir(base_dir):
                empty_image = torch.zeros((1, 64, 64, 3))
                empty_mask = torch.zeros((1, 64, 64))
                return (empty_image, empty_mask, empty_image, empty_mask, "Directory Not Found", 0, 0)

            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            try:
                state["image_files"] = sorted([
                    f for f in os.listdir(base_dir)
                    if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(base_dir, f))
                ])
                if not state["image_files"]:
                    empty_image = torch.zeros((1, 64, 64, 3))
                    empty_mask = torch.zeros((1, 64, 64))
                    return (empty_image, empty_mask, empty_image, empty_mask, "No Images Found", 0, 0)
            except Exception as e:
                empty_image = torch.zeros((1, 64, 64, 3))
                empty_mask = torch.zeros((1, 64, 64))
                return (empty_image, empty_mask, empty_image, empty_mask, f"Error listing directory: {e}", 0, 0)

        total_images = len(state["image_files"])
        state["current_index"] += 1
        if state["current_index"] >= total_images:
            state["current_index"] = 0

        if total_images == 0:
            empty_image = torch.zeros((1, 64, 64, 3))
            empty_mask = torch.zeros((1, 64, 64))
            return (empty_image, empty_mask, empty_image, empty_mask, "No Images Available", 0, 0)

        batch_images_list = []
        batch_masks_list = []
        target_size = None
        for filename in state["image_files"]:
            image_path = os.path.join(base_dir, filename)
            try:
                img = Image.open(image_path).convert("RGBA")
                if target_size is None:
                    target_size = img.size
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                
                img_np = np.array(img).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(img_np)[None,]
                
                rgb_tensor = image_tensor[:, :, :, :3]
                mask_tensor = image_tensor[:, :, :, 3]

                batch_images_list.append(rgb_tensor)
                batch_masks_list.append(mask_tensor)
            except Exception as e:
                print(f"RizzBatchImageLoader Error loading image {image_path} for batch: {e}")
        
        if not batch_images_list:
            empty_image = torch.zeros((1, 64, 64, 3))
            empty_mask = torch.zeros((1, 64, 64))
            return (empty_image, empty_mask, empty_image, empty_mask, "Failed to load any images", state["current_index"], total_images)

        image_batch_tensor = torch.cat(batch_images_list, dim=0)
        mask_batch_tensor = torch.cat(batch_masks_list, dim=0)

        current_filename = state["image_files"][state["current_index"]]
        current_image_tensor = image_batch_tensor[state["current_index"]].unsqueeze(0)
        current_mask_tensor = mask_batch_tensor[state["current_index"]].unsqueeze(0)

        return (current_image_tensor, current_mask_tensor, image_batch_tensor, mask_batch_tensor, current_filename, state["current_index"], total_images)


    @classmethod
    def IS_CHANGED(s, directory, reset_index):
        return float("NaN")

class RizzDynamicPromptGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "digital painting of a game asset concept of a medieval [subject], from a dark fantasy setting, resembling the artstyle of elden ring and dark souls, intecrite highly detailed. The background is beige."
                }),
                "subject_list": ("STRING", {
                    "multiline": True,
                    "default": "chair-lamp-torch"
                }),
            },
            "optional": {
                "reset_index": ("BOOLEAN", {"default": False, "button": True, "label": "Reset Subject Index"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT",)
    RETURN_NAMES = ("PROMPT", "CURRENT_SUBJECT", "CURRENT_INDEX", "TOTAL_SUBJECTS",)
    FUNCTION = "generate_dynamic_prompt"
    CATEGORY = "RizzNodes/Prompt"

    def generate_dynamic_prompt(self, base_prompt, subject_list, reset_index=False):
        instance_id = id(self)
        if instance_id not in _NODE_STATE:
            _NODE_STATE[instance_id] = {
                "current_subject_index": -1,
                "parsed_subject_list": [],
                "last_subject_list_raw": None,
                "last_base_prompt_raw": None
            }
        state = _NODE_STATE[instance_id]
        if reset_index or subject_list != state["last_subject_list_raw"] or base_prompt != state["last_base_prompt_raw"]:
            state["current_subject_index"] = -1
            state["last_subject_list_raw"] = subject_list
            state["last_base_prompt_raw"] = base_prompt
            state["parsed_subject_list"] = [s.strip() for s in subject_list.split('-') if s.strip()]
        total_subjects = len(state["parsed_subject_list"])
        if total_subjects == 0:
            return (base_prompt, "N/A", 0, 0)
        state["current_subject_index"] += 1
        if state["current_subject_index"] >= total_subjects:
            state["current_subject_index"] = 0
        current_subject = state["parsed_subject_list"][state["current_subject_index"]]
        generated_prompt = re.sub(r'\[subject\]', current_subject, base_prompt, flags=re.IGNORECASE)
        return (generated_prompt, current_subject, state["current_subject_index"], total_subjects)

    @classmethod
    def IS_CHANGED(s, base_prompt, subject_list, reset_index):
        return float("NaN")

class RizzModelBatchLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "input/models/glb"}),
            },
            "optional": {
                "reset_index": ("BOOLEAN", {"default": False, "button": True, "label": "Reset Index"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING", "INT", "INT",)
    RETURN_NAMES = ("MESH", "FULL_PATH_AND_FILENAME", "CURRENT_INDEX", "TOTAL_MODELS",)
    FUNCTION = "load_next_model"
    CATEGORY = "RizzNodes/Model"

    def load_next_model(self, directory, reset_index=False):
        instance_id = id(self)
        if instance_id not in _NODE_STATE:
            _NODE_STATE[instance_id] = {
                "current_model_index": -1,
                "model_files": [],
                "last_directory": None
            }
        state = _NODE_STATE[instance_id]
        if reset_index or directory != state["last_directory"]:
            state["current_model_index"] = -1
            state["last_directory"] = directory
            state["model_files"] = []
        if not state["model_files"]:
            input_dir = folder_paths.get_input_directory() if not os.path.isabs(directory) else directory
            if not os.path.isdir(input_dir):
                return (None, "Directory Not Found", 0, 0)
            valid_extensions = ('.glb', '.obj', '.gltf', '.stl', '.ply')
            try:
                state["model_files"] = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(input_dir, f))])
                if not state["model_files"]:
                    return (None, "No Models Found", 0, 0)
            except Exception as e:
                return (None, f"Error listing directory: {e}", 0, 0)
        total_models = len(state["model_files"])
        state["current_model_index"] += 1
        if state["current_model_index"] >= total_models:
            state["current_model_index"] = 0
        if total_models == 0:
            return (None, "No Models Available", 0, 0)
        current_filename = state["model_files"][state["current_model_index"]]
        model_path = os.path.join(folder_paths.get_input_directory() if not os.path.isabs(directory) else directory, current_filename)
        try:
            mesh = trimesh.load(model_path, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                if isinstance(mesh, trimesh.Scene):
                    if len(mesh.geometry) > 0:
                        mesh = sorted(mesh.geometry.values(), key=lambda g: len(g.faces), reverse=True)[0]
                    else:
                        raise ValueError("Scene contains no mesh geometry.")
            return (mesh, model_path, state["current_model_index"], total_models)
        except Exception as e:
            return (None, f"Error loading: {model_path}", state["current_model_index"], total_models)

    @classmethod
    def IS_CHANGED(s, directory, reset_index):
        return float("NaN")

class RizzUpscaleImageBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "upscale_model": ("UPSCALE_MODEL", ),
                "fixed_resolution": ("BOOLEAN", {"default": False}),
                "resolution": ([256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096],),
                "scale_method": (["LANCZOS", "NEAREST"],),
            },
            "optional": {
                "masks": ("MASK",),
                "output_type": (["BATCH", "CONCATENATED_GRID"], {"default": "BATCH"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("UPSCALED_IMAGE", "UPSCALED_MASK")
    FUNCTION = "upscale_batch"
    CATEGORY = "RizzNodes/Image"

    def upscale_batch(self, images, upscale_model, fixed_resolution, resolution, scale_method, masks=None, output_type="BATCH"):
        def _normalize_mask_batch(mask_batch):
            if mask_batch.dim() == 4:
                if mask_batch.shape[1] == 1:
                    mask_batch = mask_batch[:, 0, :, :]
                elif mask_batch.shape[-1] == 1:
                    mask_batch = mask_batch[..., 0]
            if mask_batch.dim() == 2:
                return mask_batch.unsqueeze(0)
            if mask_batch.dim() == 3:
                return mask_batch
            raise ValueError("Masks tensor must have 2, 3, or 4 dimensions.")

        def _resize_mask(mask_tensor, target_h, target_w, device):
            mask_ready = mask_tensor
            while mask_ready.dim() > 2 and mask_ready.shape[0] == 1:
                mask_ready = mask_ready.squeeze(0)
            if mask_ready.dim() > 2 and mask_ready.shape[-1] == 1:
                mask_ready = mask_ready.squeeze(-1)
            if mask_ready.dim() != 2:
                raise ValueError("Each mask must resolve to a 2D tensor (H, W).")
            mask_ready = mask_ready.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            resized = torch.nn.functional.interpolate(mask_ready, size=(target_h, target_w), mode="nearest")
            return resized[:, 0, :, :]

        def _finalize_image_output(tensor, keep_batch):
            if not keep_batch and tensor.dim() == 4 and tensor.shape[0] == 1:
                return tensor[0]
            return tensor

        def _finalize_mask_output(tensor, keep_batch):
            if not keep_batch and tensor.dim() == 3 and tensor.shape[0] == 1:
                return tensor[0]
            return tensor

        if not callable(upscale_model):
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask)

        if not isinstance(images, torch.Tensor):
            raise ValueError("Images input must be a torch.Tensor.")

        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() != 4:
            raise ValueError("Images tensor must have 3 or 4 dimensions.")

        images = images.to(dtype=torch.float32)
        num_images = images.shape[0]

        if num_images == 0:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask)

        device = images.device

        if masks is not None:
            if not isinstance(masks, torch.Tensor):
                raise ValueError("Masks input must be a torch.Tensor when provided.")
            masks = _normalize_mask_batch(masks)
            if masks.shape[0] != num_images:
                raise ValueError("Number of masks must match number of images.")
            masks = masks.to(device=device, dtype=torch.float32)

        resampling_filter = Image.LANCZOS if scale_method == "LANCZOS" else Image.NEAREST

        upscaled_images_list = []
        upscaled_masks_list = []

        pbar = comfy.utils.ProgressBar(num_images)
        for idx in range(num_images):
            image_tensor = images[idx].unsqueeze(0)
            image_tensor_permuted = image_tensor.to(dtype=torch.float32).permute(0, 3, 1, 2)
            mask_slice = masks[idx] if masks is not None else None

            try:
                upscaled = upscale_model(image_tensor_permuted)
                if not isinstance(upscaled, torch.Tensor):
                    raise ValueError("Upscale model must return a torch.Tensor.")

                if upscaled.dim() == 3:
                    upscaled = upscaled.unsqueeze(0)

                if fixed_resolution:
                    upscaled_np = upscaled.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
                    upscaled_np = np.clip(upscaled_np, 0.0, 1.0)
                    upscaled_pil = Image.fromarray((upscaled_np * 255).astype(np.uint8))
                    target_resolution = (resolution, resolution)
                    resized_pil = upscaled_pil.resize(target_resolution, resampling_filter)
                    resized_np = np.array(resized_pil).astype(np.float32) / 255.0
                    upscaled = torch.from_numpy(resized_np).unsqueeze(0).permute(0, 3, 1, 2).to(image_tensor.device)
                else:
                    upscaled = upscaled.to(image_tensor.device, dtype=torch.float32)

            except Exception as e:
                print(f"RizzUpscaleImageBatch: Error during upscale: {e}")
                upscaled = torch.zeros_like(image_tensor_permuted)
            upscaled_images_list.append(upscaled)

            target_h = upscaled.shape[2]
            target_w = upscaled.shape[3]

            if mask_slice is not None:
                try:
                    mask_resized = _resize_mask(mask_slice, target_h, target_w, image_tensor.device)
                except Exception as mask_error:
                    print(f"RizzUpscaleImageBatch: Error resizing mask: {mask_error}")
                    mask_resized = torch.ones((1, target_h, target_w), device=image_tensor.device, dtype=torch.float32)
            else:
                mask_resized = torch.ones((1, target_h, target_w), device=image_tensor.device, dtype=torch.float32)

            upscaled_masks_list.append(mask_resized)
            pbar.update(1)

        if not upscaled_images_list:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask)

        return_as_batch = (output_type == "BATCH") and (num_images > 1)

        if output_type == "CONCATENATED_GRID":
            max_h = max(img.shape[2] for img in upscaled_images_list)
            padded_images = []
            padded_masks = []
            for img in upscaled_images_list:
                pad_h = max_h - img.shape[2]
                padded_img = torch.nn.functional.pad(img, (0, 0, 0, pad_h))
                padded_images.append(padded_img)
            for mask in upscaled_masks_list:
                pad_h = max_h - mask.shape[1]
                mask_expanded = mask.unsqueeze(1)
                padded_mask = torch.nn.functional.pad(mask_expanded, (0, 0, 0, pad_h))
                padded_masks.append(padded_mask)
            final_image = torch.cat(padded_images, dim=3).permute(0, 2, 3, 1)
            final_mask = torch.cat(padded_masks, dim=3).squeeze(1)
            return (
                _finalize_image_output(final_image, keep_batch=False),
                _finalize_mask_output(final_mask, keep_batch=False),
            )

        final_images = torch.cat(upscaled_images_list, dim=0).permute(0, 2, 3, 1)
        final_masks = torch.cat(upscaled_masks_list, dim=0)

        return (
            _finalize_image_output(final_images, keep_batch=return_as_batch),
            _finalize_mask_output(final_masks, keep_batch=return_as_batch),
        )

    @classmethod
    def IS_CHANGED(s, images, upscale_model, fixed_resolution, resolution, scale_method, masks=None, output_type="BATCH"):
        return float("NaN")

class RizzBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 150.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("IMAGE", "MASK",)
    FUNCTION = "blur_masked_area"
    CATEGORY = "RizzNodes/Image"

    def blur_masked_area(self, image, strength, mask=None):
        if strength == 0:
            output_mask = mask if mask is not None else torch.ones_like(image)[:, :, :, 0]
            return (image, output_mask)

        processed_images = []
        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_pil = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8), 'RGB')
            
            blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=strength))
            
            if mask is not None:
                mask_tensor = mask[i]
                mask_pil = Image.fromarray((mask_tensor.cpu().numpy() * 255).astype(np.uint8), 'L')
                img_pil.paste(blurred_img, (0, 0), mask_pil)
                final_pil = img_pil
            else:
                final_pil = blurred_img
            
            output_np = np.array(final_pil).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_np)
            processed_images.append(output_tensor)

        final_batch = torch.stack(processed_images).to(image.device)
        
        output_mask = mask if mask is not None else torch.ones((image.shape[0], image.shape[1], image.shape[2]), device=image.device)
        
        return (final_batch, output_mask)

class RizzAlphaMargin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.2,
                    "step": 0.001
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "extend_colors"
    CATEGORY = "RizzNodes/Image"

    def extend_colors(self, image, alpha_threshold, mask=None):
        image_np = image.detach().cpu().numpy().astype(np.float32, copy=False)
        device = image.device

        if mask is not None:
            mask_np = mask.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            if image_np.shape[-1] == 4:
                mask_np = image_np[..., 3]
                image_np = image_np[..., :3]
            else:
                h, w = image_np.shape[1:3]
                mask_np = np.ones((image_np.shape[0], h, w), dtype=np.float32)

        if mask_np.ndim == 4:
            mask_np = np.squeeze(mask_np, axis=-1)

        processed = []
        for i in range(image_np.shape[0]):
            filled = self._extend_single(image_np[i, :, :, :3], mask_np[i], alpha_threshold)
            processed.append(torch.from_numpy(filled))

        output_image = torch.stack(processed).to(device)
        output_mask = torch.from_numpy(mask_np.astype(np.float32)).to(device)
        return (output_image, output_mask)

    @staticmethod
    def _extend_single(rgb, alpha_mask, alpha_threshold):
        rgb_data = rgb.copy().astype(np.float32, copy=False)
        mask = (alpha_mask > alpha_threshold)

        if not np.any(mask):
            return rgb_data
        if np.all(mask):
            return rgb_data

        missing = ~mask
        _, indices = scipy.ndimage.distance_transform_edt(missing, return_indices=True)

        nearest_y = indices[0][missing]
        nearest_x = indices[1][missing]

        for channel in range(rgb_data.shape[-1]):
            channel_data = rgb_data[:, :, channel]
            channel_data[missing] = channel_data[nearest_y, nearest_x]

        return rgb_data

class RizzCropAndScaleFromMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
                "interpolation": (["bicubic", "bilinear", "nearest"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "CROP_DATA")
    RETURN_NAMES = ("CROPPED_IMAGES", "original_image", "CROP_DATA")
    FUNCTION = "crop_and_scale"
    CATEGORY = "RizzNodes/Image/Transform"

    def crop_and_scale(self, image, mask, target_width, target_height, padding, interpolation):
        batch_size, img_height, img_width, _ = image.shape
        mask_np = mask.cpu().numpy()
        image_np = image.cpu().numpy()
        
        cropped_images_tensors = []
        crop_data = []

        pil_interpolation = {
            "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST
        }[interpolation]
        
        for i in range(batch_size):
            single_mask = mask_np[i]
            labeled_mask, num_features = scipy.ndimage.label(single_mask)
            
            if num_features == 0:
                continue

            slices = scipy.ndimage.find_objects(labeled_mask)
            for slc in slices:
                y_slice, x_slice = slc
                
                y1, y2 = y_slice.start, y_slice.stop
                x1, x2 = x_slice.start, x_slice.stop

                y1_pad = max(0, y1 - padding)
                y2_pad = min(img_height, y2 + padding)
                x1_pad = max(0, x1 - padding)
                x2_pad = min(img_width, x2 + padding)

                crop_info = {
                    "original_bbox": [x1_pad, y1_pad, x2_pad, y2_pad],
                    "original_img_idx": i
                }
                crop_data.append(crop_info)

                cropped_np = image_np[i, y1_pad:y2_pad, x1_pad:x2_pad, :]
                cropped_pil = Image.fromarray((cropped_np * 255).astype(np.uint8))
                
                scaled_pil = cropped_pil.resize((target_width, target_height), pil_interpolation)
                
                scaled_np = np.array(scaled_pil).astype(np.float32) / 255.0
                scaled_tensor = torch.from_numpy(scaled_np)
                cropped_images_tensors.append(scaled_tensor)

        if not cropped_images_tensors:
            return (torch.zeros((1, target_height, target_width, 3)), image, [])

        final_batch = torch.stack(cropped_images_tensors).to(image.device)
        return (final_batch, image, crop_data)


class RizzPasteAndUnscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_images": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "feathering": ("INT", {"default": 64, "min": 0, "max": 512, "step": 1}),
                "interpolation": (["bicubic", "bilinear", "nearest"],),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_and_unscale"
    CATEGORY = "RizzNodes/Image/Transform"

    def paste_and_unscale(self, original_image, processed_images, crop_data, feathering, interpolation):
        if not crop_data:
            return (original_image,)
        
        composited_image = original_image.clone()
        pil_interpolation = {
            "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST
        }[interpolation]

        for i, data in enumerate(crop_data):
            x1, y1, x2, y2 = data["original_bbox"]
            original_idx = data["original_img_idx"]
            
            w, h = x2 - x1, y2 - y1
            
            processed_img_tensor = processed_images[i]
            processed_pil = Image.fromarray((processed_img_tensor.cpu().numpy() * 255).astype(np.uint8))
            
            unscaled_pil = processed_pil.resize((w, h), pil_interpolation)
            unscaled_tensor = torch.from_numpy(np.array(unscaled_pil).astype(np.float32) / 255.0).to(original_image.device)
            
            if feathering > 0:
                mask = torch.ones(h, w)
                feather_amount = min(feathering, h // 2, w // 2)
                if feather_amount > 0:
                    y_ramp = torch.linspace(0, 1, feather_amount)
                    x_ramp = torch.linspace(0, 1, feather_amount)
                    mask[:feather_amount, :] *= y_ramp.unsqueeze(1)
                    mask[-feather_amount:, :] *= y_ramp.flip(0).unsqueeze(1)
                    mask[:, :feather_amount] *= x_ramp.unsqueeze(0)
                    mask[:, -feather_amount:] *= x_ramp.flip(0).unsqueeze(0)

                mask = mask.unsqueeze(-1).to(original_image.device)
                original_chunk = composited_image[original_idx, y1:y2, x1:x2, :]
                blended_chunk = original_chunk * (1 - mask) + unscaled_tensor * mask
                composited_image[original_idx, y1:y2, x1:x2, :] = blended_chunk
            else:
                composited_image[original_idx, y1:y2, x1:x2, :] = unscaled_tensor
        
        return (composited_image,)

class RizzClean:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "purge_vram"
    CATEGORY = "RizzNodes/Utilities"

    def purge_vram(self, anything, purge_cache, purge_models):
        if purge_models:
            mm.unload_all_models()

        if purge_cache:
            mm.soft_empty_cache()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()

        gc.collect()
        return (anything,)

class RizzEditImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                "flip_mode": (["none", "horizontal", "vertical", "both"], {"default": "none"}),
                "invert_colors": ("BOOLEAN", {"default": False}),
                "rotate_image": ("BOOLEAN", {"default": False}),
                "rotation_angle": (["90", "180", "270"], {"default": "90"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "RizzNodes/Image"

    def edit_image(
        self,
        image,
        brightness,
        contrast,
        hue,
        saturation,
        flip_horizontal=False,
        flip_vertical=False,
        flip_mode="none",
        invert_colors=False,
        rotate_image=False,
        rotation_angle="90",
    ):
        normalized_flip_mode = flip_mode.lower() if isinstance(flip_mode, str) else "none"
        apply_flip_horizontal = flip_horizontal
        apply_flip_vertical = flip_vertical

        if normalized_flip_mode == "horizontal":
            apply_flip_horizontal, apply_flip_vertical = True, False
        elif normalized_flip_mode == "vertical":
            apply_flip_horizontal, apply_flip_vertical = False, True
        elif normalized_flip_mode == "both":
            apply_flip_horizontal, apply_flip_vertical = True, True

        rotation_steps = {"90": 1, "180": 2, "270": 3}
        rotation_key = str(rotation_angle)
        rotate_times = rotation_steps.get(rotation_key, 0)

        processed_images = []
        for img_tensor in image:
            img_np = img_tensor.cpu().numpy()

            if brightness != 1.0 or contrast != 1.0:
                img_np = img_np * brightness
                img_np = (img_np - 0.5) * contrast + 0.5
                img_np = np.clip(img_np, 0.0, 1.0)

            if hue != 0.0 or saturation != 1.0:
                hsv = rgb_to_hsv(img_np)
                hsv[..., 0] = (hsv[..., 0] + hue / 360.0) % 1.0
                hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
                img_np = hsv_to_rgb(hsv)

            if invert_colors:
                img_np = 1.0 - img_np

            if apply_flip_horizontal:
                img_np = np.flip(img_np, axis=1)

            if apply_flip_vertical:
                img_np = np.flip(img_np, axis=0)

            if rotate_image and rotate_times:
                img_np = np.rot90(img_np, k=rotate_times)

            img_np = np.ascontiguousarray(img_np)
            processed_images.append(torch.from_numpy(img_np))

        final_batch = torch.stack(processed_images).to(image.device)
        return (final_batch,)

class RizzChannelPack:
    @classmethod
    def INPUT_TYPES(cls):
        resolutions = ["auto", 128, 256, 512, 1024, 2048, 4096]
        return {
            "required": {},
            "optional": {
                "R_Image": ("IMAGE",),
                "G_Image": ("IMAGE",),
                "B_Image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "resolution": (resolutions, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "pack_channels"
    CATEGORY = "RizzNodes/Image"

    def pack_channels(self, R_Image=None, G_Image=None, B_Image=None, upscale_model=None, resolution="auto"):
        ref_image = next((img for img in [R_Image, G_Image, B_Image] if img is not None), None)

        if resolution == "auto":
            if ref_image is not None:
                _, target_res, _, _ = ref_image.shape
            else:
                target_res = 512 
            target_h, target_w = target_res, target_res
        else:
            target_h, target_w = resolution, resolution

        if ref_image is None:
            packed_image = torch.zeros((1, target_h, target_w, 3), dtype=torch.float32)
        else:
            device = ref_image.device
            black_image = torch.zeros((1, target_h, target_w, 3), dtype=torch.float32, device=device)

            if R_Image is None: R_Image = black_image
            if G_Image is None: G_Image = black_image
            if B_Image is None: B_Image = black_image

            images_to_process = [R_Image, G_Image, B_Image]
            resized_images = []

            for img in images_to_process:
                if img.shape[1] != target_h or img.shape[2] != target_w:
                    img_permuted = img.permute(0, 3, 1, 2)
                    img_resized_permuted = torch.nn.functional.interpolate(img_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    resized_images.append(img_resized_permuted.permute(0, 2, 3, 1))
                else:
                    resized_images.append(img)

            r_channel = resized_images[0][..., 0]
            g_channel = resized_images[1][..., 0]
            b_channel = resized_images[2][..., 0]
            packed_image = torch.stack([r_channel, g_channel, b_channel], dim=-1)

        if upscale_model is not None:
            img_permuted = packed_image.permute(0, 3, 1, 2)
            upscaled_permuted = upscale_model(img_permuted)
            packed_image = upscaled_permuted.permute(0, 2, 3, 1)

        return (packed_image,)

class RizzChannelSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("R_Channel", "G_Channel", "B_Channel")
    FUNCTION = "split_channels"
    CATEGORY = "RizzNodes/Image"

    def split_channels(self, image, upscale_model=None):
        processed_image = image

        if upscale_model is not None and callable(upscale_model):
            try:
                img_permuted = image.permute(0, 3, 1, 2)
                upscaled_permuted = upscale_model(img_permuted)
                processed_image = upscaled_permuted.permute(0, 2, 3, 1)
            except Exception as e:
                print(f"RizzChannelSplit: Error during upscale: {e}")
                processed_image = image
        
        r_channel_data = processed_image[..., 0]
        g_channel_data = processed_image[..., 1]
        b_channel_data = processed_image[..., 2]

        r_image = torch.stack([r_channel_data, r_channel_data, r_channel_data], dim=-1)
        g_image = torch.stack([g_channel_data, g_channel_data, g_channel_data], dim=-1)
        b_image = torch.stack([b_channel_data, b_channel_data, b_channel_data], dim=-1)

        return (r_image, g_image, b_image,)

def _to_ml_mesh(mesh: trimesh.Trimesh) -> ml.Mesh:
    return ml.Mesh(
        vertex_matrix=mesh.vertices.astype(np.float64),
        face_matrix=mesh.faces.astype(np.int32)
    )

def _from_ml_mesh(mesh: ml.Mesh) -> trimesh.Trimesh:
    v = np.array(mesh.vertex_matrix(), dtype=np.float32)
    f = np.array(mesh.face_matrix(), dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)

class SimplifyMeshNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "target_faces": ("INT", {
                    "default": 80000,
                    "min": 1000,
                    "max": 10_000_000,
                    "step": 1000
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("simplified_mesh",)
    FUNCTION = "simplify"
    CATEGORY = "Mesh/Processing"

    def simplify(self, mesh, target_faces):
        face_num = len(mesh.faces)
        if face_num <= target_faces:
            print(f"[SimplifyMeshNode] No simplification: {face_num} faces <= {target_faces}")
            return (mesh.copy(),)

        ms = ml.MeshSet()
        ms.add_mesh(_to_ml_mesh(mesh), "input")

        for f in ("meshing_remove_duplicate_faces", "meshing_remove_duplicate_vertices", "meshing_remove_unreferenced_vertices"):
            try:
                ms.apply_filter(f)
            except Exception:
                pass

        try:
            print("[SimplifyMeshNode] Merging close vertices")
            ms.apply_filter("meshing_merge_close_vertices")
        except Exception as e:
            print(f"[SimplifyMeshNode] merge close vertices failed: {e}")

        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=int(target_faces),
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarweight=0.3,
            qualitythr=0.5,
            boundaryweight=1.0,
        )

        simplified = _from_ml_mesh(ms.current_mesh())
        print(f"[SimplifyMeshNode] Simplified {face_num} → {len(simplified.faces)} faces (target {target_faces})")
        return (simplified,)

NODE_CLASS_MAPPINGS = {
    "RizzLoadLatestImage": RizzLoadLatestImage,
    "RizzLoadLatestMesh": RizzLoadLatestMesh,
    "RizzBatchImageLoader": RizzBatchImageLoader,
    "RizzDynamicPromptGenerator": RizzDynamicPromptGenerator,
    "RizzModelBatchLoader": RizzModelBatchLoader,
    "RizzUpscaleImageBatch": RizzUpscaleImageBatch,
    "RizzBlur": RizzBlur,
    "RizzAlphaMargin": RizzAlphaMargin,
    "RizzCropAndScaleFromMask": RizzCropAndScaleFromMask,
    "RizzPasteAndUnscale": RizzPasteAndUnscale,
    "RizzClean": RizzClean,
    "RizzEditImage": RizzEditImage,
    "RizzChannelPack": RizzChannelPack,
    "RizzChannelSplit": RizzChannelSplit,
    "SimplifyMesh": SimplifyMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RizzLoadLatestImage": "Load Latest Image (Rizz)",
    "RizzLoadLatestMesh": "Load Latest Mesh (Rizz)",
    "RizzBatchImageLoader": "Batch Image Loader",
    "RizzDynamicPromptGenerator": "Dynamic Prompt Generator",
    "RizzModelBatchLoader": "Batch Model Loader",
    "RizzUpscaleImageBatch": "Batch Upscale Image",
    "RizzBlur": "Blur (Masked)",
    "RizzAlphaMargin": "Alpha Margin Fill",
    "RizzCropAndScaleFromMask": "Crop & Scale from Mask",
    "RizzPasteAndUnscale": "Paste & Unscale",
    "RizzClean": "Memory Cleaner",
    "RizzEditImage": "Edit Image (Brightness/Contrast/Hue/Saturation)",
    "RizzChannelPack": "Channel Pack (Rizz)",
    "RizzChannelSplit": "Channel Split (Rizz)",
    "SimplifyMesh": "Simplify Mesh (PyMeshLab)",
}
