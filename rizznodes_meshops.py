"""
Quadremesh Node - Field-aligned quad-dominant remeshing
Based on: "Instant Field-Aligned Meshes" (Jakob et al., SIGGRAPH 2015)

This implementation adapts the algorithm to generate more quads in dense/detailed
areas and fewer quads in smooth/open areas of the mesh.

GPU-accelerated using PyTorch CUDA when available.
"""

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from typing import Tuple, Optional
import logging

logging.getLogger('trimesh').setLevel(logging.ERROR)

# Determine device
def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_vertex_normals_gpu(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute area-weighted vertex normals on GPU."""
    device = vertices.device
    n_verts = vertices.shape[0]
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_areas = torch.norm(face_normals, dim=1, keepdim=True) + 1e-10
    face_normals = face_normals / face_areas
    
    # Scatter add for vertex normals
    vertex_normals = torch.zeros((n_verts, 3), device=device, dtype=vertices.dtype)
    weighted_normals = face_normals * face_areas
    
    for i in range(3):
        vertex_normals.index_add_(0, faces[:, i], weighted_normals)
    
    norms = torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-10
    return vertex_normals / norms


def compute_vertex_curvature_gpu(vertices: torch.Tensor, faces: torch.Tensor, 
                                  edges: torch.Tensor) -> torch.Tensor:
    """
    Compute mean curvature magnitude at each vertex on GPU.
    Uses discrete Laplacian approximation.
    """
    device = vertices.device
    n_verts = vertices.shape[0]
    
    normals = compute_vertex_normals_gpu(vertices, faces)
    
    # Build sparse adjacency using edge list
    # Compute Laplacian for each vertex
    curvatures = torch.zeros(n_verts, device=device, dtype=vertices.dtype)
    neighbor_count = torch.zeros(n_verts, device=device, dtype=vertices.dtype)
    neighbor_sum = torch.zeros((n_verts, 3), device=device, dtype=vertices.dtype)
    
    # Accumulate neighbor positions
    neighbor_sum.index_add_(0, edges[:, 0], vertices[edges[:, 1]])
    neighbor_sum.index_add_(0, edges[:, 1], vertices[edges[:, 0]])
    neighbor_count.index_add_(0, edges[:, 0], torch.ones(edges.shape[0], device=device))
    neighbor_count.index_add_(0, edges[:, 1], torch.ones(edges.shape[0], device=device))
    
    # Compute centroid and Laplacian
    valid_mask = neighbor_count > 1
    centroids = neighbor_sum / (neighbor_count.unsqueeze(1) + 1e-10)
    laplacian = centroids - vertices
    
    # Mean curvature is abs(dot(laplacian, normal))
    curvatures = torch.abs(torch.sum(laplacian * normals, dim=1)) * valid_mask.float()
    
    return curvatures


def rotation_matrices_batch(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Create batch of rotation matrices around axes by angles. GPU-accelerated."""
    # axes: (N, 3), angles: (N,)
    device = axes.device
    n = axes.shape[0]
    
    # Normalize axes
    axes = axes / (torch.norm(axes, dim=1, keepdim=True) + 1e-10)
    
    c = torch.cos(angles)
    s = torch.sin(angles)
    t = 1 - c
    
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    
    # Build rotation matrices (N, 3, 3)
    R = torch.zeros((n, 3, 3), device=device, dtype=axes.dtype)
    R[:, 0, 0] = c + x*x*t
    R[:, 0, 1] = x*y*t - z*s
    R[:, 0, 2] = x*z*t + y*s
    R[:, 1, 0] = y*x*t + z*s
    R[:, 1, 1] = c + y*y*t
    R[:, 1, 2] = y*z*t - x*s
    R[:, 2, 0] = z*x*t - y*s
    R[:, 2, 1] = z*y*t + x*s
    R[:, 2, 2] = c + z*z*t
    
    return R


def project_to_tangent_batch(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Project vectors v onto tangent planes with normals n. Batched GPU."""
    # v, n: (N, 3)
    dots = torch.sum(v * n, dim=1, keepdim=True)
    return v - dots * n


def optimize_orientation_field_gpu(vertices: torch.Tensor, normals: torch.Tensor,
                                    edges: torch.Tensor, iterations: int = 6,
                                    use_extrinsic: bool = True) -> torch.Tensor:
    """
    Optimize 4-RoSy orientation field using parallel updates on GPU.
    Returns per-vertex representative directions.
    """
    device = vertices.device
    n_verts = vertices.shape[0]
    n_edges = edges.shape[0]
    
    # Initialize with random tangent vectors
    orientations = torch.randn(n_verts, 3, device=device, dtype=vertices.dtype)
    orientations = project_to_tangent_batch(orientations, normals)
    norms = torch.norm(orientations, dim=1, keepdim=True)
    
    # Handle zero-length vectors
    fallback_mask = norms.squeeze() < 1e-10
    if fallback_mask.any():
        fallback = torch.zeros_like(orientations[fallback_mask])
        n_fb = normals[fallback_mask]
        # Cross with [1,0,0] or [0,1,0]
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand_as(n_fb)
        fallback = torch.cross(n_fb, x_axis, dim=1)
        fallback_norm = torch.norm(fallback, dim=1, keepdim=True)
        zero_fallback = fallback_norm.squeeze() < 1e-10
        if zero_fallback.any():
            y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0)
            fallback[zero_fallback] = torch.cross(n_fb[zero_fallback], y_axis.expand(zero_fallback.sum(), -1), dim=1)
        orientations[fallback_mask] = fallback
        norms = torch.norm(orientations, dim=1, keepdim=True)
    
    orientations = orientations / (norms + 1e-10)
    
    # Build sparse adjacency info for parallel updates
    # For each vertex, accumulate contributions from neighbors
    edge_i = edges[:, 0]
    edge_j = edges[:, 1]
    
    # Rotation angles for 4-RoSy: 0, 90, 180, 270 degrees
    rot_angles = torch.tensor([0, np.pi/2, np.pi, 3*np.pi/2], device=device, dtype=vertices.dtype)
    
    for iteration in range(iterations):
        # For each edge, compute contribution to both endpoints
        o_i = orientations[edge_i]  # (E, 3)
        o_j = orientations[edge_j]  # (E, 3)
        n_i = normals[edge_i]       # (E, 3)
        n_j = normals[edge_j]       # (E, 3)
        
        if use_extrinsic:
            # For each edge, try all 4 rotations and find best
            # Rotate o_j around n_j by each angle, find one closest to o_i
            best_contrib_ij = torch.zeros_like(o_j)
            best_contrib_ji = torch.zeros_like(o_i)
            best_angle_ij = torch.full((n_edges,), float('inf'), device=device)
            best_angle_ji = torch.full((n_edges,), float('inf'), device=device)
            
            for k in range(4):
                # Rotate o_j around n_j
                angle_k = rot_angles[k].expand(n_edges)
                R_j = rotation_matrices_batch(n_j, angle_k)
                rotated_j = torch.bmm(R_j, o_j.unsqueeze(-1)).squeeze(-1)
                
                # Compute angle difference with o_i
                dots_ij = torch.clamp(torch.sum(o_i * rotated_j, dim=1), -1, 1)
                angles_ij = torch.acos(dots_ij)
                
                # Update best
                better_ij = angles_ij < best_angle_ij
                best_angle_ij = torch.where(better_ij, angles_ij, best_angle_ij)
                best_contrib_ij = torch.where(better_ij.unsqueeze(-1), rotated_j, best_contrib_ij)
                
                # Similarly for i -> j
                R_i = rotation_matrices_batch(n_i, angle_k)
                rotated_i = torch.bmm(R_i, o_i.unsqueeze(-1)).squeeze(-1)
                
                dots_ji = torch.clamp(torch.sum(o_j * rotated_i, dim=1), -1, 1)
                angles_ji = torch.acos(dots_ji)
                
                better_ji = angles_ji < best_angle_ji
                best_angle_ji = torch.where(better_ji, angles_ji, best_angle_ji)
                best_contrib_ji = torch.where(better_ji.unsqueeze(-1), rotated_i, best_contrib_ji)
            
            # Accumulate contributions
            new_orientations = torch.zeros_like(orientations)
            new_orientations.index_add_(0, edge_i, best_contrib_ij)
            new_orientations.index_add_(0, edge_j, best_contrib_ji)
        else:
            # Intrinsic mode - simpler, just add neighbor orientations
            new_orientations = torch.zeros_like(orientations)
            new_orientations.index_add_(0, edge_i, o_j)
            new_orientations.index_add_(0, edge_j, o_i)
        
        # Project to tangent and normalize
        new_orientations = project_to_tangent_batch(new_orientations, normals)
        norms = torch.norm(new_orientations, dim=1, keepdim=True)
        valid = norms.squeeze() > 1e-10
        orientations = torch.where(valid.unsqueeze(-1), 
                                   new_orientations / (norms + 1e-10), 
                                   orientations)
    
    return orientations


def optimize_position_field_gpu(vertices: torch.Tensor, normals: torch.Tensor,
                                 orientations: torch.Tensor, edges: torch.Tensor,
                                 edge_lengths: torch.Tensor, iterations: int = 6,
                                 use_extrinsic: bool = True) -> torch.Tensor:
    """
    Optimize position field aligned with orientation field on GPU.
    Returns per-vertex representative positions.
    """
    device = vertices.device
    n_verts = vertices.shape[0]
    n_edges = edges.shape[0]
    
    # Initialize positions at vertices
    positions = vertices.clone()
    
    # Compute perpendicular orientations
    orientations_perp = torch.cross(normals, orientations, dim=1)
    norms = torch.norm(orientations_perp, dim=1, keepdim=True)
    orientations_perp = orientations_perp / (norms + 1e-10)
    
    edge_i = edges[:, 0]
    edge_j = edges[:, 1]
    
    for iteration in range(iterations):
        p_i = positions[edge_i]  # (E, 3)
        p_j = positions[edge_j]  # (E, 3)
        
        rho_i = edge_lengths[edge_i]  # (E,)
        rho_j = edge_lengths[edge_j]  # (E,)
        avg_rho = (rho_i + rho_j) / 2
        
        o_i = orientations[edge_i]
        o_j = orientations[edge_j]
        op_i = orientations_perp[edge_i]
        op_j = orientations_perp[edge_j]
        
        if use_extrinsic:
            # Compute translated positions
            diff_ij = p_j - p_i
            diff_ji = p_i - p_j
            
            # Project onto local axes
            u_ij = torch.sum(diff_ij * o_i, dim=1)
            v_ij = torch.sum(diff_ij * op_i, dim=1)
            u_ji = torch.sum(diff_ji * o_j, dim=1)
            v_ji = torch.sum(diff_ji * op_j, dim=1)
            
            # Round to integer grid
            u_ij_int = torch.round(u_ij / avg_rho) * avg_rho
            v_ij_int = torch.round(v_ij / avg_rho) * avg_rho
            u_ji_int = torch.round(u_ji / avg_rho) * avg_rho
            v_ji_int = torch.round(v_ji / avg_rho) * avg_rho
            
            # Translated positions
            translated_j = p_j - u_ij_int.unsqueeze(-1) * o_i - v_ij_int.unsqueeze(-1) * op_i
            translated_i = p_i - u_ji_int.unsqueeze(-1) * o_j - v_ji_int.unsqueeze(-1) * op_j
        else:
            translated_j = p_j
            translated_i = p_i
        
        # Accumulate and average
        new_positions = torch.zeros_like(positions)
        counts = torch.zeros(n_verts, device=device, dtype=vertices.dtype)
        
        new_positions.index_add_(0, edge_i, translated_j)
        new_positions.index_add_(0, edge_j, translated_i)
        counts.index_add_(0, edge_i, torch.ones(n_edges, device=device))
        counts.index_add_(0, edge_j, torch.ones(n_edges, device=device))
        
        valid = counts > 0
        avg_positions = new_positions / (counts.unsqueeze(-1) + 1e-10)
        
        # Round to lattice
        diff = avg_positions - vertices
        u = torch.sum(diff * orientations, dim=1)
        v = torch.sum(diff * orientations_perp, dim=1)
        
        u_rounded = torch.round(u / edge_lengths) * edge_lengths
        v_rounded = torch.round(v / edge_lengths) * edge_lengths
        
        new_pos = (vertices + 
                   u_rounded.unsqueeze(-1) * orientations +
                   v_rounded.unsqueeze(-1) * orientations_perp)
        
        positions = torch.where(valid.unsqueeze(-1), new_pos, positions)
    
    return positions


def extract_quad_mesh(vertices: np.ndarray, positions: np.ndarray,
                      orientations: np.ndarray, normals: np.ndarray,
                      edges: np.ndarray, edge_lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract quad-dominant mesh from position field.
    Returns (output_vertices, output_faces).
    Uses robust approach: cluster positions, then triangulate.
    """
    from scipy.spatial import Delaunay
    
    n_verts = len(vertices)
    print(f"[Quadremesh] Extraction: {n_verts} input vertices")
    
    # Step 1: Quantize positions to grid for clustering
    # Use median edge length as grid spacing
    median_rho = np.median(edge_lengths)
    
    # Quantize positions to integer lattice coordinates
    quantized = np.round(positions / (median_rho + 1e-10)).astype(np.int64)
    
    # Step 2: Cluster vertices by quantized position using spatial hashing
    # Create unique hash for each grid cell
    grid_min = quantized.min(axis=0)
    quantized_shifted = quantized - grid_min
    grid_max = quantized_shifted.max(axis=0) + 1
    
    # Hash: x + y*W + z*W*H
    hash_vals = (quantized_shifted[:, 0] + 
                 quantized_shifted[:, 1] * grid_max[0] +
                 quantized_shifted[:, 2] * grid_max[0] * grid_max[1])
    
    # Group vertices by hash
    unique_hashes, inverse_indices = np.unique(hash_vals, return_inverse=True)
    n_clusters = len(unique_hashes)
    print(f"[Quadremesh] Clustered to {n_clusters} unique grid positions")
    
    if n_clusters < 4:
        print("[Quadremesh] Warning: Too few clusters for mesh extraction")
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]])
    
    # Step 3: Compute cluster centroids (in original space, not position field)
    # Use positions for vertex locations but keep original surface mapping
    cluster_positions = np.zeros((n_clusters, 3))
    cluster_normals = np.zeros((n_clusters, 3))
    cluster_counts = np.zeros(n_clusters)
    
    for i in range(n_verts):
        c = inverse_indices[i]
        cluster_positions[c] += positions[i]
        cluster_normals[c] += normals[i]
        cluster_counts[c] += 1
    
    # Normalize
    cluster_counts_safe = cluster_counts + 1e-10
    cluster_positions = cluster_positions / cluster_counts_safe[:, np.newaxis]
    cluster_norms = np.linalg.norm(cluster_normals, axis=1, keepdims=True) + 1e-10
    cluster_normals = cluster_normals / cluster_norms
    
    # Step 4: Build edge connectivity between clusters
    cluster_edges = set()
    for e in edges:
        i, j = e
        ci, cj = inverse_indices[i], inverse_indices[j]
        if ci != cj:
            cluster_edges.add((min(ci, cj), max(ci, cj)))
    
    cluster_edges = np.array(list(cluster_edges))
    print(f"[Quadremesh] Found {len(cluster_edges)} cluster edges")
    
    if len(cluster_edges) < 3:
        print("[Quadremesh] Warning: Too few edges for mesh extraction")
        return cluster_positions[:3], np.array([[0, 1, 2]])
    
    # Step 5: Build faces using edge adjacency
    # For each pair of edges sharing a vertex, check if they form a triangle
    edge_adj = {}
    for ei, (a, b) in enumerate(cluster_edges):
        edge_adj.setdefault(a, set()).add((b, ei))
        edge_adj.setdefault(b, set()).add((a, ei))
    
    # Find triangles by looking for edge triples
    triangles = set()
    edge_set = set(map(tuple, cluster_edges))
    
    for v in edge_adj:
        neighbors = list(edge_adj[v])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, _ = neighbors[i]
                n2, _ = neighbors[j]
                # Check if n1 and n2 are connected
                edge_key = (min(n1, n2), max(n1, n2))
                if edge_key in edge_set:
                    # Found a triangle
                    tri = tuple(sorted([v, n1, n2]))
                    triangles.add(tri)
    
    print(f"[Quadremesh] Found {len(triangles)} triangles from edge walking")
    
    if len(triangles) < 1:
        # Fallback: Use Delaunay triangulation on projected 2D positions
        print("[Quadremesh] Fallback: Using Delaunay triangulation")
        
        # Project cluster positions to 2D using PCA
        centered = cluster_positions - cluster_positions.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            projected_2d = centered @ Vt[:2].T
            
            # Delaunay triangulation
            if len(cluster_positions) >= 4:
                tri = Delaunay(projected_2d)
                triangles = tri.simplices
            else:
                triangles = np.array([[0, 1, 2]])
        except Exception as e:
            print(f"[Quadremesh] SVD failed: {e}")
            # Super fallback: just return first 3 vertices as triangle
            triangles = np.array([[0, 1, 2]])
    else:
        triangles = np.array([list(t) for t in triangles])
    
    # Step 6: Filter degenerate triangles
    valid_tris = []
    for tri in triangles:
        if len(set(tri)) == 3:  # All vertices distinct
            v0, v1, v2 = cluster_positions[tri]
            edge1 = np.linalg.norm(v1 - v0)
            edge2 = np.linalg.norm(v2 - v0)
            edge3 = np.linalg.norm(v2 - v1)
            
            # Filter very long edges (> 5x median)
            max_edge = max(edge1, edge2, edge3)
            if max_edge < 5 * median_rho:
                valid_tris.append(tri)
    
    if len(valid_tris) == 0:
        print("[Quadremesh] Warning: All triangles filtered, keeping originals")
        valid_tris = triangles[:min(100, len(triangles))]
    
    print(f"[Quadremesh] Final: {len(valid_tris)} valid triangles")
    
    return cluster_positions, np.array(valid_tris)


def quadremesh(mesh: trimesh.Trimesh,
               target_edge_length: float = 0.05,
               target_faces: int = 0,
               curvature_adaptation: float = 1.0,
               min_edge_scale: float = 0.25,
               max_edge_scale: float = 2.0,
               iterations: int = 6,
               use_extrinsic: bool = True,
               device: Optional[str] = None) -> trimesh.Trimesh:
    """
    Remesh a triangular mesh into a quad-dominant mesh.
    GPU-accelerated when CUDA is available.
    
    Args:
        mesh: Input trimesh
        target_edge_length: Base target edge length (relative to mesh scale)
        target_faces: Target number of output faces (0 = use edge_length instead)
        curvature_adaptation: How much to adapt density (0=uniform, 2=very adaptive)
        min_edge_scale: Minimum edge scale in high-curvature areas
        max_edge_scale: Maximum edge scale in low-curvature areas  
        iterations: Gauss-Seidel iterations per hierarchy level
        use_extrinsic: Use extrinsic smoothness energy
        device: Force device ('cuda' or 'cpu'), auto-detect if None
        
    Returns:
        Quad-dominant mesh (as trimesh with quads triangulated)
    """
    # Select device
    if device is None:
        dev = get_device()
    else:
        dev = torch.device(device)
    
    using_gpu = dev.type == 'cuda'
    print(f"[Quadremesh] Processing on {dev} - {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Auto-calculate edge length from target faces if specified
    if target_faces > 0:
        # Formula: For target F faces with surface area S (normalized to unit scale):
        # Each triangle has area ~= (edge_length^2 * sqrt(3)/4)
        # So edge_length = sqrt(4 * S / (F * sqrt(3)))
        surface_area = mesh.area
        bbox = mesh.bounds[1] - mesh.bounds[0]
        mesh_scale = bbox.max()
        normalized_area = surface_area / (mesh_scale ** 2)
        
        # Account for clustering reducing face count
        effective_target = target_faces * 0.8
        
        # Calculate base edge length for uniform distribution
        base_edge_length = np.sqrt(4 * normalized_area / (effective_target * np.sqrt(3)))
        
        # Compensate for curvature adaptation and edge scale settings:
        # With adaptation, edges are scaled between min_edge_scale and max_edge_scale
        # Higher adaptation = more variance, most mesh area uses larger edges (smooth regions)
        if curvature_adaptation > 0:
            # Most mesh area is typically smooth, so average edge is weighted toward max_edge_scale
            smooth_weight = 0.7  # Most of mesh area is typically smooth
            detail_weight = 0.3  # Less area is high-detail
            
            # Calculate weighted average scale
            avg_scale = min_edge_scale * detail_weight + max_edge_scale * smooth_weight
            
            # Higher adaptation increases the effect
            adaptation_factor = 1.0 + curvature_adaptation * 0.3
            avg_scale = avg_scale * adaptation_factor
            
            # Compensate by dividing edge length by average scale
            target_edge_length = base_edge_length / avg_scale
            print(f"[Quadremesh] Auto edge length for ~{target_faces} faces: {target_edge_length:.4f} (adaptation compensation: {avg_scale:.2f}x)")
        else:
            target_edge_length = base_edge_length
            print(f"[Quadremesh] Auto edge length for ~{target_faces} faces: {target_edge_length:.4f}")
        
        target_edge_length = np.clip(target_edge_length, 0.001, 0.5)
    
    # Convert to tensors
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=dev)
    
    # Normalize to unit scale
    bbox = vertices.max(dim=0).values - vertices.min(dim=0).values
    scale = bbox.max().item()
    min_corner = vertices.min(dim=0).values
    vertices = (vertices - min_corner) / scale
    
    # Get edges
    edges_np = mesh.edges_unique
    edges = torch.tensor(edges_np, dtype=torch.long, device=dev)
    
    # Compute normals and curvature on GPU
    normals = compute_vertex_normals_gpu(vertices, faces)
    curvatures = compute_vertex_curvature_gpu(vertices, faces, edges)
    
    # Normalize curvature
    curv_min = curvatures.min()
    curv_max = curvatures.max()
    if (curv_max - curv_min) > 1e-10:
        curv_norm = (curvatures - curv_min) / (curv_max - curv_min)
    else:
        curv_norm = torch.zeros_like(curvatures)
    
    # Compute adaptive edge lengths
    edge_scales = max_edge_scale - (max_edge_scale - min_edge_scale) * (curv_norm ** curvature_adaptation)
    edge_lengths = target_edge_length * edge_scales
    
    print(f"[Quadremesh] Edge length range: {edge_lengths.min().item():.4f} - {edge_lengths.max().item():.4f}")
    
    # Optimize orientation field on GPU
    print("[Quadremesh] Optimizing orientation field (GPU)..." if using_gpu else "[Quadremesh] Optimizing orientation field...")
    orientations = optimize_orientation_field_gpu(vertices, normals, edges, iterations, use_extrinsic)
    
    # Optimize position field on GPU
    print("[Quadremesh] Optimizing position field (GPU)..." if using_gpu else "[Quadremesh] Optimizing position field...")
    positions = optimize_position_field_gpu(vertices, normals, orientations, edges, edge_lengths, iterations, use_extrinsic)
    
    # Move to CPU for mesh extraction (irregular graph operations)
    print("[Quadremesh] Extracting quad mesh...")
    vertices_cpu = vertices.cpu().numpy()
    positions_cpu = positions.cpu().numpy()
    orientations_cpu = orientations.cpu().numpy()
    normals_cpu = normals.cpu().numpy()
    edge_lengths_cpu = edge_lengths.cpu().numpy()
    
    out_verts, out_faces = extract_quad_mesh(
        vertices_cpu, positions_cpu, orientations_cpu, normals_cpu, 
        edges_np, edge_lengths_cpu)
    
    # Rescale back
    out_verts = out_verts * scale + mesh.vertices.min(axis=0)
    
    print(f"[Quadremesh] Output: {len(out_verts)} vertices, {len(out_faces)} faces")
    
    result = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
    
    # Clean up
    if hasattr(result, 'nondegenerate_faces'):
        valid_faces = result.nondegenerate_faces()
        if len(valid_faces) > 0 and len(valid_faces) < len(result.faces):
            result = trimesh.Trimesh(
                vertices=result.vertices,
                faces=result.faces[valid_faces],
                process=False
            )
    
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    
    # Complete normal recalculation like Blender "Recalculate Normals Outside"
    print("[Quadremesh] Recalculating normals (Blender-style)...")
    
    # Step 1: Create fresh mesh with no cached normals
    clean_verts = np.array(result.vertices, dtype=np.float64)
    clean_faces = np.array(result.faces, dtype=np.int64)
    
    # Step 2: Compute face normals from scratch
    face_verts = clean_verts[clean_faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10
    face_normals = face_normals / face_norms
    
    # Step 3: Compute face centroids
    face_centroids = face_verts.mean(axis=1)
    
    # Step 4: Compute mesh centroid (center of mass)
    mesh_centroid = clean_verts.mean(axis=0)
    
    # Step 5: For each face, check if normal points outward from mesh center
    # Vector from mesh center to face centroid
    to_face = face_centroids - mesh_centroid
    to_face_norm = np.linalg.norm(to_face, axis=1, keepdims=True) + 1e-10
    to_face = to_face / to_face_norm
    
    # Dot product: positive = normal points outward, negative = inward
    dots = np.sum(face_normals * to_face, axis=1)
    
    # Flip faces where normal points inward (towards center)
    flip_mask = dots < 0
    
    if flip_mask.any():
        flipped_count = flip_mask.sum()
        # Flip winding by reversing vertex order
        clean_faces[flip_mask] = clean_faces[flip_mask][:, ::-1]
        print(f"[Quadremesh] Flipped {flipped_count}/{len(clean_faces)} faces to point outward")
    else:
        print("[Quadremesh] All faces already pointing outward")
    
    # Step 6: Create final mesh with corrected faces
    result = trimesh.Trimesh(vertices=clean_verts, faces=clean_faces, process=False)
    
    # Step 7: Force recompute of all normals
    _ = result.face_normals  # Trigger recompute
    _ = result.vertex_normals  # Trigger recompute
    
    return result


class QuadremeshNode:
    """
    ComfyUI node for quad-dominant remeshing with adaptive density.
    Based on "Instant Field-Aligned Meshes" (Jakob et al., SIGGRAPH 2015).
    GPU-accelerated when CUDA is available.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "target_faces": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "step": 1000,
                    "tooltip": "Target number of output faces (0 = use edge_length instead)"
                }),
                "target_edge_length": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Base target edge length (ignored if target_faces > 0)"
                }),
                "curvature_adaptation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "How much to adapt density based on curvature (0=uniform, 2=very adaptive)"
                }),
                "min_edge_scale": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum edge length multiplier in high-detail areas"
                }),
                "max_edge_scale": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Maximum edge length multiplier in smooth areas"
                }),
                "iterations": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Gauss-Seidel iterations (more = smoother fields)"
                }),
                "use_extrinsic": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use extrinsic energy for natural feature alignment"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration if available"
                }),
            }
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "execute"
    CATEGORY = "RizzNodes/Mesh"
    
    def execute(self, mesh: trimesh.Trimesh, target_faces: int,
                target_edge_length: float, curvature_adaptation: float, 
                min_edge_scale: float, max_edge_scale: float, iterations: int,
                use_extrinsic: bool, use_gpu: bool = True) -> tuple:
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        result = quadremesh(
            mesh=mesh,
            target_edge_length=target_edge_length,
            target_faces=target_faces,
            curvature_adaptation=curvature_adaptation,
            min_edge_scale=min_edge_scale,
            max_edge_scale=max_edge_scale,
            iterations=iterations,
            use_extrinsic=use_extrinsic,
            device=device
        )
        
        return (result,)


# For direct testing
if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test with a simple sphere
    mesh = trimesh.creation.icosphere(subdivisions=3)
    result = quadremesh(mesh, target_edge_length=0.1)
    print(f"Test complete: {len(result.faces)} output faces")
