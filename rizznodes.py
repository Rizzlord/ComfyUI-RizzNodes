import os
import torch
from PIL import Image
import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery
import logging
import re
import gc
import comfy.model_management as mm
import folder_paths

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

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "CURRENT_FILENAME", "CURRENT_INDEX", "TOTAL_IMAGES",)
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
        if not state["image_files"] or directory != state["last_directory"]:
            input_dir = folder_paths.get_input_directory() if not os.path.isabs(directory) else directory
            if not os.path.isdir(input_dir):
                print(f"RizzBatchImageLoader Error: Directory not found - {input_dir}")
                return (torch.zeros(64, 64, 3), "Directory Not Found", 0, 0)
            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            try:
                state["image_files"] = sorted([
                    f for f in os.listdir(input_dir)
                    if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(input_dir, f))
                ])
                if not state["image_files"]:
                    print(f"RizzBatchImageLoader Warning: No valid image files found in {input_dir}")
                    return (torch.zeros(64, 64, 3), "No Images Found", 0, 0)
            except Exception as e:
                print(f"RizzBatchImageLoader Error listing directory {input_dir}: {e}")
                return (torch.zeros(64, 64, 3), f"Error listing directory: {e}", 0, 0)
        total_images = len(state["image_files"])
        state["current_index"] += 1
        if state["current_index"] >= total_images:
            state["current_index"] = 0
            print(f"RizzBatchImageLoader: Looping back to the first image for instance {instance_id}")
        if total_images == 0:
            print(f"RizzBatchImageLoader Warning: No images to load after index update for instance {instance_id}")
            return (torch.zeros(64, 64, 3), "No Images Available", 0, 0)
        current_filename = state["image_files"][state["current_index"]]
        image_path = os.path.join(folder_paths.get_input_directory() if not os.path.isabs(directory) else directory, current_filename)
        try:
            img = Image.open(image_path).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(img)[None,]
            state["current_image_data"] = image_tensor
            print(f"RizzBatchImageLoader: Loaded image {current_filename} (Index: {state['current_index']}/{total_images-1}) for instance {instance_id}")
            return (image_tensor, current_filename, state["current_index"], total_images)
        except Exception as e:
            print(f"RizzBatchImageLoader Error loading image {image_path}: {e}")
            return (torch.zeros(64, 64, 3), f"Error loading: {current_filename}", state["current_index"], total_images)

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
    "RizzClean": "Memory Cleaner",
    "RizzAnything": "Anything Passthrough (Reroute)",
}