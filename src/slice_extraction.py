import numpy as np
from scipy.ndimage import map_coordinates


def extract_slice(reference, annotation, anchoring_matrix, height, width, thickness_voxels=10):
    """
    Extracts an oblique slice from the 3D reference and annotation volumes.

    Parameters:
        reference (ndarray): 3D reference volume (e.g., template).
        annotation (ndarray): 3D annotation volume (e.g., labeled atlas).
        anchoring_matrix (ndarray): 3x3 matrix [origin, u-axis, v-axis] from DeepSlice.
        height (int): Output slice height (pixels).
        width (int): Output slice width (pixels).
        thickness_voxels (int): Thickness for the projection mask in voxels.

    Returns:
        Tuple:
            - 2D reference slice (float32)
            - 2D annotation slice (uint16)
            - 3D binary mask used for slice projection (uint8)
    """
    o, u, v = anchoring_matrix

    # 🔁 Orthonormalize u and v
    u_hat = u / np.linalg.norm(u)
    v_proj = v - np.dot(v, u_hat) * u_hat
    v_hat = v_proj / np.linalg.norm(v_proj)

    # 🧮 Create pixel-centered 2D grid
    i_grid, j_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # 🧠 Compute 3D coordinates for each pixel
    coords_3d = (
        o[None, None, :] +
        j_grid[..., None] * (u_hat / width * np.linalg.norm(u)) +
        i_grid[..., None] * (v_hat / height * np.linalg.norm(v_proj))
    )
    coords_flat = coords_3d.reshape(-1, 3).T  # shape (3, N)

    # 🎯 Sample 3D reference and annotation using interpolation
    slice_sampled_ref = map_coordinates(reference, coords_flat, order=1, mode='constant', cval=0)
    slice_img_ref = slice_sampled_ref.reshape(height, width)

    slice_sampled_anno = map_coordinates(annotation, coords_flat, order=0, mode='constant', cval=0)
    slice_img_anno = slice_sampled_anno.reshape(height, width)

    # 🧩 Create 3D projection mask (thickness-aware)
    mask_3d = np.zeros_like(reference, dtype=np.uint8)
    coords_vox = np.round(coords_flat).astype(int)
    valid_mask = np.all((coords_vox >= 0) & (coords_vox < np.array(reference.shape)[:, None]), axis=0)
    coords_vox = coords_vox[:, valid_mask]

    # 🔺 Offset points along the slice normal to generate a 3D slab
    normal = np.cross(u_hat, v_hat)
    normal /= np.linalg.norm(normal)
    half_thickness = thickness_voxels // 2
    offsets = np.arange(-half_thickness, half_thickness + 1)

    for offset in offsets:
        offset_coords = coords_vox + np.round(normal[:, None] * offset).astype(int)
        valid = np.all((offset_coords >= 0) & (offset_coords < np.array(reference.shape)[:, None]), axis=0)
        shifted_coords = offset_coords[:, valid]
        mask_3d[shifted_coords[0], shifted_coords[1], shifted_coords[2]] = 1

    return slice_img_ref.astype(np.float32), slice_img_anno.astype(np.uint16), mask_3d


def project_points(points_3d, anchoring_matrix, width, height, thickness_voxels=10):
    """
    Projects 3D coordinates into a 2D slice plane defined by an anchoring matrix.

    Parameters:
        points_3d (ndarray): N x 3 array of 3D coordinates.
        anchoring_matrix (ndarray): 3x3 matrix [origin, u-axis, v-axis].
        width (int): Slice width in pixels.
        height (int): Slice height in pixels.
        thickness_voxels (int): Thickness of the slice in voxels.

    Returns:
        Tuple:
            - points_2d (N x 2): Projected 2D points inside the slice
            - mask (bool array): Boolean mask for selected 3D points in the slab
            - in_bounds (bool array): Boolean mask for points inside image frame
    """
    o, u, v = anchoring_matrix

    # 🔁 Orthonormalize u and v
    u_hat = u / np.linalg.norm(u)
    v_proj = v - np.dot(v, u_hat) * u_hat
    v_hat = v_proj / np.linalg.norm(v_proj)

    # 🧮 Compute normal vector to the slice
    normal = np.cross(u_hat, v_hat)

    # 🔍 Select points close to the slice plane (within thickness)
    dists = np.dot(points_3d - o, normal)
    mask = np.abs(dists) <= (thickness_voxels / 2)
    slab_points = points_3d[mask]

    if slab_points.shape[0] == 0:
        return np.empty((0, 2)), mask, np.array([], dtype=bool)

    # 📐 Project onto u-v axes
    x_proj = np.dot(slab_points - o, u_hat)
    y_proj = np.dot(slab_points - o, v_hat)

    # 🎯 Scale to pixel coordinates
    j = x_proj * width / np.linalg.norm(u)
    i = y_proj * height / np.linalg.norm(v_proj)
    points_2d = np.stack([i, j], axis=1)

    # 🧱 Check if inside image bounds
    in_bounds = (i >= 0) & (i < height) & (j >= 0) & (j < width)

    return points_2d[in_bounds], mask, in_bounds

