import torch
import torch.nn.functional as F
import numpy as np
import random

def simple_tile(image, overlap=0.25):
    """
    Makes an image tileable by blending edges.
    image: (B, H, W, C) tensor
    overlap: fraction of image to blend (0.0-0.5)
    """
    B, H, W, C = image.shape
    
    # We will blend the edges into the center
    # Shift image so edges are now in the center
    shift_h = H // 2
    shift_w = W // 2
    
    shifted = torch.roll(image, shifts=(shift_h, shift_w), dims=(1, 2))
    
    # Create mask for blending
    # 0 at center (where original edges are), 1 at new edges
    
    # Ensure overlap is valid
    overlap = max(0.01, min(0.49, overlap))
    
    h_blend = int(H * overlap)
    w_blend = int(W * overlap)
    
    # Linear blend masks
    mask_h = torch.ones((1, H, 1, 1), device=image.device)
    mask_w = torch.ones((1, 1, W, 1), device=image.device)
    
    # Horizontal smoothstep
    lin_h = torch.linspace(0, 1, h_blend, device=image.device)
    smooth_h = 3*lin_h**2 - 2*lin_h**3 # Smoothstep
    
    # Vertical smoothstep (width dimension)
    lin_w = torch.linspace(0, 1, w_blend, device=image.device)
    smooth_w = 3*lin_w**2 - 2*lin_w**3
    
    # Center region [shift_h - h_blend : shift_h + h_blend] is where the seam is
    # We want to use the shifted image, but blend it with itself? 
    # Actually, simpler approach:
    # 1. Tile the image 2x2
    # 2. Crop a center HxW
    # 3. Blend the seams
    
    # Better approach for "Make Seamless":
    # Cross-fade the image with itself offset by half size
    
    layer1 = image
    layer2 = torch.roll(image, shifts=(shift_h, shift_w), dims=(1, 2))
    
    # Radial mask fading from center to corners?
    # Or just linear gradient blend
    
    # Let's do the standard "offset and blend" technique efficiently
    # Center of layer2 contains the seam of layer1
    
    # Mask: 1 in center, 0 at edges.
    # But layer2 has seam at edges? No, layer2 is pure at edges.
    
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing="ij"
    )
    
    # Distance from center
    dist = torch.sqrt(X*X + Y*Y)
    dist = torch.clamp(dist, 0, 1)
    
    # Invert dist so it's 1 at center, 0 at corners
    mask = 1.0 - dist
    # Contrast adjustment to sharp blend
    mask = torch.clamp((mask - 0.5) / (2 * overlap) + 0.5, 0, 1)
    mask = mask.unsqueeze(0).unsqueeze(-1)
    
    # This simple radial blend often looks decent for textures
    return layer1 * mask + layer2 * (1 - mask)


def random_rotation_matrix(angle_min, angle_max, batch_size, device):
    angles = (torch.rand(batch_size, device=device) * (angle_max - angle_min) + angle_min)
    c = torch.cos(angles)
    s = torch.sin(angles)
    # 2x3 affine matrices
    row1 = torch.stack([c, -s, torch.zeros_like(c)], dim=1) 
    row2 = torch.stack([s, c, torch.zeros_like(c)], dim=1)
    return torch.stack([row1, row2], dim=1) # (B, 2, 3)

def voronoi_tile(image, scale=1.0, overlap=0.25, random_rotate=True, seed=0):
    """
    Advanced stochastic tiling using something similar to 'Texture Bombing' 
    but efficient on GPU.
    """
    B, H, W, C = image.shape
    device = image.device
    
    torch.manual_seed(seed)
    
    # Grid size determines feature density
    # Scale=1.0 -> Grid cells are approx image size
    # Scale=2.0 -> Grid cells are half image size (more density)
    
    # We want to cover the output output canvas (H, W) with patches
    # To do this efficiently, we define a grid where each cell contributes a splat.
    
    grid_h = max(2, int(scale * 2))
    grid_w = max(2, int(scale * 2 * (W/H)))
    
    # Grid of feature points
    # Each cell has a random offset from its center
    
    # Generate grid coordinates
    ys = torch.linspace(-1, 1, grid_h + 1, device=device)[:-1] + (1/grid_h)
    xs = torch.linspace(-1, 1, grid_w + 1, device=device)[:-1] + (1/grid_w)
    
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y], dim=-1) # (Gh, Gw, 2)
    
    # Jitter points
    jitter_amp = 0.8 / min(grid_h, grid_w)
    noise = (torch.rand_like(grid_points) * 2 - 1) * jitter_amp
    centers = grid_points + noise
    centers = centers.view(-1, 2) # (N_cells, 2)
    
    # For each pixel in output, find closest N centers? Too slow.
    # Instead, splash each patch onto the canvas.
    
    output = torch.zeros(B, C, H, W, device=device)
    weight_acc = torch.zeros(B, 1, H, W, device=device)
    
    # Because we loop over cells, keep grid size reasonable (< 50 cells)
    num_cells = centers.shape[0]
    
    # Prepare input image for sampling (B, C, H, W)
    img_nchw = image.permute(0, 3, 1, 2)
    
    # Pre-generate random params for all cells
    if random_rotate:
        rotations = (torch.rand(num_cells, device=device) * 2 * 3.14159)
    else:
        rotations = torch.zeros(num_cells, device=device)
        
    random_scales = torch.rand(num_cells, device=device) * 0.4 + 0.8 # 0.8 to 1.2
    
    # Create coordinate grid for the CANVAS
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )
    # (H, W, 2)
    coords = torch.stack([X, Y], dim=-1).unsqueeze(0).expand(num_cells, -1, -1, -1) 
    
    # Process in chunks if too many cells to save VRAM
    # But for specialized texture node, simple loop is fine.
    
    # We need to splash the SAME image onto the canvas at different centers.
    # Inverse: For each pixel, which source pixel does it map to for a given cell?
    # Source = Rotation * (Pixel - Center) / Scale
    
    # Let's vectorize over cells? (Num_cells * H * W) might be big (e.g. 20 * 1024*1024 * 4 bytes = 80MB. Safe.)
    
    # Centers: (N, 1, 1, 2)
    c = centers.view(num_cells, 1, 1, 2)
    
    # Translate coords so center is 0
    rel_coords = coords - c
    
    # Wrap to range [-1, 1] for seamless tiling (toroidal topology)
    # The domain is size 2 (-1 to 1).
    rel_coords = rel_coords - 2.0 * torch.round(rel_coords / 2.0)
    
    # Rotate
    cos_r = torch.cos(-rotations).view(num_cells, 1, 1) # Inverse rotation for sampling
    sin_r = torch.sin(-rotations).view(num_cells, 1, 1)
    
    rot_x = rel_coords[..., 0] * cos_r - rel_coords[..., 1] * sin_r
    rot_y = rel_coords[..., 0] * sin_r + rel_coords[..., 1] * cos_r
    
    # Scale
    s = random_scales.view(num_cells, 1, 1)
    sample_coords_x = rot_x / s
    sample_coords_y = rot_y / s
    
    # Stack for grid_sample (N, H, W, 2)
    curr_grid = torch.stack([sample_coords_x, sample_coords_y], dim=-1)
    
    # Compute Weights based on distance to center (Soft Voronoi)
    # Distance in output space
    d2 = torch.sum(rel_coords ** 2, dim=-1) # (N, H, W)
    # Gaussian falloff
    # Radius depends on grid density
    radius = 2.5 / min(grid_h, grid_w)
    weights = torch.exp(-d2 / (radius**2 * overlap))
    
    # Sample image
    # We expand image to (N*B, C, H, W) - wait, B is batch size of images. 
    # Usually B=1. If B>1, we might need to handle per-image distinct tiling.
    # For now, presume B=1 or share grid across batch.
    
    # Repeat image for each cell
    # img: (B, C, H, W)
    # We want to sample this image using curr_grid which is (N, H, W, 2)
    # If B=1:
    sampled = F.grid_sample(
        img_nchw.expand(num_cells, -1, -1, -1),
        curr_grid,
        mode='bilinear',
        padding_mode='reflection', # tiling!
        align_corners=False
    )
    # sampled: (N, C, H, W)
    
    # Weighted sum
    w = weights.unsqueeze(1) # (N, 1, H, W)
    weighted_patches = sampled * w
    
    # Accumulate weighted colour (va)
    va = torch.sum(weighted_patches, dim=0) # (C, H, W)
    
    # Weight sum (w1)
    w1 = torch.sum(w, dim=0) + 1e-5
    
    # Weight squared sum (w2) for contrast preservation
    w2 = torch.sum(w * w, dim=0) + 1e-5
    
    # Standard tiling result (mean)
    mean_result = va / w1
    
    # Contrast preservation: 
    # Formula: res = contrast + (va - w1 * contrast) / sqrt(w2)
    # Here 'contrast' refers to the mean color of the source texture
    
    # Calculate source mean color
    source_mean = torch.mean(img_nchw, dim=(2, 3)).squeeze().view(-1, 1, 1) # (C, 1, 1)
    
    # Apply contrast formula
    # (va - w1 * source_mean) / sqrt(w2) + source_mean
    variance_result = (va - w1 * source_mean) / torch.sqrt(w2) + source_mean

    # Blend between mean and variance result based on overlap? 
    # Actually the article implies this IS the result.
    # However, variance preserving blending can sometimes be harsh.
    # Let's provide a mix or just use it.
    # Given the user specifically asked "did you do this", let's use the formula.
    
    # But let's actally do a mix to be safe, or just return variance_result.
    # The article says: float4 col = mix(va / w1, res, offset);
    # Where offset seems to be an interpolation factor.
    # Let's assume we want full effect or maybe scaled by overlap? 
    # For now, let's output the high-quality result.
    
    final_img = variance_result
    
    # final_img is (C, H, W)
    
    # Add batch dim back: (1, C, H, W)
    if B == 1:
        final_img = final_img.unsqueeze(0)
        
    # Rearrange to (B, H, W, C) for ComfyUI
    return final_img.permute(0, 2, 3, 1)


class RizzMakeTileable:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["simple", "voronoi"],),
                "overlap": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "random_rotate": ("BOOLEAN", {"default": True}),
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_tileable"
    CATEGORY = "RizzNodes/Texture"

    def make_tileable(self, image, method, overlap, tile_scale, random_rotate, variation_seed):
        # image is (B, H, W, C)
        
        output_images = []
        
        # Process each image in batch independently if needed, 
        # but my optimized functions handle B=1. For B>1 loop.
        for i in range(image.shape[0]):
            img = image[i:i+1] # (1, H, W, C)
            
            if method == "simple":
                # Regular offset blend
                out = simple_tile(img, overlap)
            else:
                # Stochastic Voronoi
                out = voronoi_tile(img, tile_scale, overlap, random_rotate, variation_seed)
            
            output_images.append(out)
        
        return (torch.cat(output_images, dim=0),)
        

# Import necessary modules for preview
import folder_paths
import random
import os
from PIL import Image
import numpy as np

class RizzPreviewTiling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    # RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "RizzNodes/Texture"
    OUTPUT_NODE = True

    def execute(self, image, prompt=None, extra_pnginfo=None):
        # Save image implementation similar to generic PreviewImage
        # But also returns the image for pass-through
        
        # 1. Convert to PIL
        results = []
        for tensor in image:
            array = 255. * tensor.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            
            # Save to temp folder
            filename_prefix = "RizzPreview"
            file = f"{filename_prefix}_{random.randint(1, 10000000)}.png"
            subfolder = "rizz_previews"
            
            full_output_folder = os.path.join(folder_paths.get_temp_directory(), subfolder)
            os.makedirs(full_output_folder, exist_ok=True)
            
            img.save(os.path.join(full_output_folder, file))
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "temp"
            })

        return {"ui": {"images": results}}
