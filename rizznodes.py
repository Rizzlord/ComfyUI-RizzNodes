import os
import torch
from PIL import Image, ImageFilter
import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery
import logging
import re
import gc
import comfy.model_management as mm
import folder_paths
import scipy.ndimage

logging.getLogger('trimesh').setLevel(logging.ERROR)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

_NODE_STATE = {}


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
            print(f"RizzLoadLatestImage Error: Directory not found - {search_path}")
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

                print(f"RizzLoadLatestImage: Successfully loaded latest image '{latest_file}' from '{search_path}'")
                return (image_rgb, mask, latest_file)
            except Exception as e:
                print(f"RizzLoadLatestImage Error: Failed to load image '{file_path}': {e}")
                return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)), f"Error loading: {e}")
        else:
            print(f"RizzLoadLatestImage Warning: No matching image found for base name '{filename_base}' in '{search_path}'")
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
            print(f"RizzLoadLatestMesh Error: Directory not found - {search_path}")
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
                print(f"RizzLoadLatestMesh: Successfully loaded latest mesh '{latest_file}' from '{search_path}'")
                return (mesh, latest_file)
            except Exception as e:
                print(f"RizzLoadLatestMesh Error: Failed to load mesh '{file_path}': {e}")
                return (trimesh.Trimesh(), f"Error loading: {e}")
        else:
            print(f"RizzLoadLatestMesh Warning: No matching mesh found for base name '{filename_base}' in '{search_path}'")
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT",)
    RETURN_NAMES = ("CURRENT_IMAGE", "IMAGE_BATCH", "CURRENT_FILENAME", "CURRENT_INDEX", "TOTAL_IMAGES",)
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
            print(f"RizzBatchImageLoader: Resetting index for instance {instance_id}")

        base_dir = folder_paths.get_input_directory() if not os.path.isabs(directory) else directory
        
        if not state["image_files"]:
            if not os.path.isdir(base_dir):
                print(f"RizzBatchImageLoader Error: Directory not found - {base_dir}")
                empty_tensor = torch.zeros((1, 64, 64, 3))
                return (empty_tensor, empty_tensor, "Directory Not Found", 0, 0)

            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            try:
                state["image_files"] = sorted([
                    f for f in os.listdir(base_dir)
                    if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(base_dir, f))
                ])
                if not state["image_files"]:
                    print(f"RizzBatchImageLoader Warning: No valid image files found in {base_dir}")
                    empty_tensor = torch.zeros((1, 64, 64, 3))
                    return (empty_tensor, empty_tensor, "No Images Found", 0, 0)
            except Exception as e:
                print(f"RizzBatchImageLoader Error listing directory {base_dir}: {e}")
                empty_tensor = torch.zeros((1, 64, 64, 3))
                return (empty_tensor, empty_tensor, f"Error listing directory: {e}", 0, 0)

        total_images = len(state["image_files"])
        state["current_index"] += 1
        if state["current_index"] >= total_images:
            state["current_index"] = 0
            print(f"RizzBatchImageLoader: Looping back to the first image for instance {instance_id}")

        if total_images == 0:
            empty_tensor = torch.zeros((1, 64, 64, 3))
            return (empty_tensor, empty_tensor, "No Images Available", 0, 0)

        batch_images_list = []
        for filename in state["image_files"]:
            image_path = os.path.join(base_dir, filename)
            try:
                img = Image.open(image_path).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(img_np)[None,]
                batch_images_list.append(image_tensor)
            except Exception as e:
                print(f"RizzBatchImageLoader Error loading image {image_path} for batch: {e}")
        
        image_batch_tensor = torch.cat(batch_images_list, dim=0) if batch_images_list else torch.zeros((1, 64, 64, 3))

        current_filename = state["image_files"][state["current_index"]]
        current_image_path = os.path.join(base_dir, current_filename)
        try:
            current_img_pil = Image.open(current_image_path).convert("RGB")
            current_img_np = np.array(current_img_pil).astype(np.float32) / 255.0
            current_image_tensor = torch.from_numpy(current_img_np)[None,]
            
            print(f"RizzBatchImageLoader: Loaded image {current_filename} (Index: {state['current_index']}/{total_images-1}) for instance {instance_id}")
            return (current_image_tensor, image_batch_tensor, current_filename, state["current_index"], total_images)
        except Exception as e:
            print(f"RizzBatchImageLoader Error loading current image {current_image_path}: {e}")
            empty_tensor = torch.zeros((1, 64, 64, 3))
            return (empty_tensor, image_batch_tensor, f"Error loading: {current_filename}", state["current_index"], total_images)


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
            print(f"RizzDynamicPromptGenerator: Initialized new state for instance {instance_id}")
        state = _NODE_STATE[instance_id]
        if reset_index or subject_list != state["last_subject_list_raw"] or base_prompt != state["last_base_prompt_raw"]:
            state["current_subject_index"] = -1
            state["last_subject_list_raw"] = subject_list
            state["last_base_prompt_raw"] = base_prompt
            state["parsed_subject_list"] = [s.strip() for s in subject_list.split('-') if s.strip()]
            print(f"RizzDynamicPromptGenerator: RESETTING index for instance {instance_id}. Found {len(state['parsed_subject_list'])} subjects.")
        total_subjects = len(state["parsed_subject_list"])
        if total_subjects == 0:
            print(f"RizzDynamicPromptGenerator Warning: Subject list is empty for instance {instance_id}")
            return (base_prompt, "N/A", 0, 0)
        state["current_subject_index"] += 1
        if state["current_subject_index"] >= total_subjects:
            state["current_subject_index"] = 0
            print(f"RizzDynamicPromptGenerator: Looping back to the first subject for instance {instance_id}")
        current_subject = state["parsed_subject_list"][state["current_subject_index"]]
        generated_prompt = re.sub(r'\[subject\]', current_subject, base_prompt, flags=re.IGNORECASE)
        print(f"RizzDynamicPromptGenerator (Instance: {instance_id}): "
              f"Index {state['current_subject_index']}/{total_subjects-1} -> "
              f"Subject: '{current_subject}'")
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
            print(f"RizzModelBatchLoader: Resetting index for instance {instance_id}")
        if not state["model_files"]:
            input_dir = folder_paths.get_input_directory() if not os.path.isabs(directory) else directory
            if not os.path.isdir(input_dir):
                print(f"RizzModelBatchLoader Error: Directory not found - {input_dir}")
                return (None, "Directory Not Found", 0, 0)
            valid_extensions = ('.glb', '.obj', '.gltf', '.stl', '.ply')
            try:
                state["model_files"] = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(input_dir, f))])
                if not state["model_files"]:
                    print(f"RizzModelBatchLoader Warning: No valid model files found in {input_dir}")
                    return (None, "No Models Found", 0, 0)
            except Exception as e:
                print(f"RizzModelBatchLoader Error listing directory {input_dir}: {e}")
                return (None, f"Error listing directory: {e}", 0, 0)
        total_models = len(state["model_files"])
        state["current_model_index"] += 1
        if state["current_model_index"] >= total_models:
            state["current_model_index"] = 0
            print(f"RizzModelBatchLoader: Looping back to the first model for instance {instance_id}")
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
            print(f"RizzModelBatchLoader: Loaded model {current_filename} (Index: {state['current_model_index']}/{total_models-1})")
            return (mesh, model_path, state["current_model_index"], total_models)
        except Exception as e:
            print(f"RizzModelBatchLoader Error loading model {model_path}: {e}")
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
            },
            "optional": {
                "output_type": (["BATCH", "CONCATENATED_GRID"], {"default": "BATCH"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("UPSCALE_IMAGES",)
    FUNCTION = "upscale_batch"
    CATEGORY = "RizzNodes/Image"

    def upscale_batch(self, images, upscale_model, output_type):
        upscaled_images_list = []
        if not callable(upscale_model):
            print("RizzUpscaleImageBatch ERROR: upscale_model is not callable.")
            return (torch.zeros(64, 64, 3)[None,],)
        for i, image_tensor in enumerate(images):
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=torch.float32).permute(0, 3, 1, 2)
            try:
                upscaled = upscale_model(image_tensor)
                upscaled_images_list.append(upscaled)
                print(f"RizzUpscaleImageBatch: Upscaled image {i+1}/{len(images)}")
            except Exception as e:
                print(f"RizzUpscaleImageBatch ERROR on image {i}: {e}")
                upscaled_images_list.append(torch.zeros_like(image_tensor))
        if not upscaled_images_list:
            return (torch.zeros(64, 64, 3)[None,],)
        if output_type == "CONCATENATED_GRID":
            first_h = upscaled_images_list[0].shape[1]
            if all(img.shape[1] == first_h for img in upscaled_images_list):
                final_output = torch.cat(upscaled_images_list, dim=2)
            else:
                max_w = max(img.shape[2] for img in upscaled_images_list)
                padded = [torch.nn.functional.pad(img, (0, max_w - img.shape[2], 0, 0)) for img in upscaled_images_list]
                final_output = torch.cat(padded, dim=1)
            return (final_output,)
        final_output = torch.cat(upscaled_images_list, dim=0).permute(0, 2, 3, 1)
        return (final_output,)

    @classmethod
    def IS_CHANGED(s, images, upscale_model, output_type):
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
    def INPUT_TYPES(s):
        return { "required": { "trigger": (any,), } }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "clean_memory"
    CATEGORY = "RizzNodes/Utilities"

    def clean_memory(self, trigger):
        print("RizzClean: Starting memory cleanup...")
        print(" - Unloading all models...")
        mm.unload_all_models()
        print(" - Clearing cache...")
        mm.soft_empty_cache()
        print(" - Collecting garbage...")
        gc.collect()
        print("RizzClean: Memory cleanup complete.")
        return (trigger,)

    @classmethod
    def IS_CHANGED(s, trigger):
        return float("NaN")

class RizzAnything:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "anything": (any,), } }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "passthrough"
    CATEGORY = "RizzNodes/Utilities"

    def passthrough(self, anything):
        return (anything,)

    @classmethod
    def IS_CHANGED(s, anything):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "RizzLoadLatestImage": RizzLoadLatestImage,
    "RizzLoadLatestMesh": RizzLoadLatestMesh,
    "RizzBatchImageLoader": RizzBatchImageLoader,
    "RizzDynamicPromptGenerator": RizzDynamicPromptGenerator,
    "RizzModelBatchLoader": RizzModelBatchLoader,
    "RizzUpscaleImageBatch": RizzUpscaleImageBatch,
    "RizzBlur": RizzBlur,
    "RizzCropAndScaleFromMask": RizzCropAndScaleFromMask,
    "RizzPasteAndUnscale": RizzPasteAndUnscale,
    "RizzClean": RizzClean,
    "RizzAnything": RizzAnything,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RizzLoadLatestImage": "Load Latest Image (Rizz)",
    "RizzLoadLatestMesh": "Load Latest Mesh (Rizz)",
    "RizzBatchImageLoader": "Batch Image Loader",
    "RizzDynamicPromptGenerator": "Dynamic Prompt Generator",
    "RizzModelBatchLoader": "Batch Model Loader",
    "RizzUpscaleImageBatch": "Batch Upscale Image",
    "RizzBlur": "Blur (Masked)",
    "RizzCropAndScaleFromMask": "Crop & Scale from Mask",
    "RizzPasteAndUnscale": "Paste & Unscale",
    "RizzClean": "Memory Cleaner",
    "RizzAnything": "Anything Passthrough (Reroute)",
}