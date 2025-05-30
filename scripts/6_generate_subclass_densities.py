import os
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter
from natsort import natsorted

from src.utils import load_yaml_config

# --- Load config ---
config = load_yaml_config("config/config.yml")

working_dir = config.get("working_dir", "")
resources_dir = config.get("resources_dir", "resources/atlas")
metadata_path = config.get("st_metadata_file", "aba_combined_data.parquet")
reference_path = config.get("reference_template_file", "aba_template_mouse.tif")
saving_dir = os.path.join(working_dir, config.get("subclass_kde_output_dir", "data/subclass_kde"))
bandwidth_mm = config.get("kde_bandwidth_mm", 0.2)
voxel_size_mm = config.get("voxel_size_mm", 0.025)
density_downsampling = config.get("density_downsampling", 4)

os.makedirs(saving_dir, exist_ok=True)

# --- Load resources ---
st_metadata = pd.read_parquet(os.path.join(resources_dir, metadata_path))
reference = tifffile.imread(os.path.join(resources_dir, reference_path))
reference_shape = reference.shape
grid_shape = np.array(reference_shape) // density_downsampling
blur_sigma = (bandwidth_mm / density_downsampling) / voxel_size_mm  # Adjust for voxel size

z_max = reference_shape[0]
subclasses = natsorted(st_metadata["subclass"].unique())

# --- Main loop: Generate KDE per subclass ---
for subclass in subclasses:
    subclass_safe = subclass.replace("/", "-")
    mask = st_metadata["subclass"] == subclass
    coords = st_metadata.loc[mask, ["z", "x", "y"]].values

    if coords.shape[0] < 10:
        print(f"⚠️ Skipping {subclass_safe} — too few points")
        continue

    print(f"🔍 Processing {subclass_safe} with {coords.shape[0]} cells...")

    # Mirror across Z
    coords_mirror = coords.copy()
    coords_mirror[:, 0] = z_max - coords[:, 0] - 1
    coords_augmented = np.vstack([coords, coords_mirror])

    # Convert to voxel indices
    coords_ds = coords_augmented / density_downsampling
    coords_vox = np.round(coords_ds).astype(int)

    # Fill density volume
    density = np.zeros(grid_shape, dtype=np.float32)
    for z, x, y in coords_vox:
        if 0 <= z < grid_shape[0] and 0 <= x < grid_shape[1] and 0 <= y < grid_shape[2]:
            density[z, x, y] += 1

    # Gaussian smoothing
    density = gaussian_filter(density, sigma=blur_sigma)

    # Flip to match visualization orientation
    density = np.flip(density, 1)
    density = np.flip(density, 2)

    # Save result
    out_path = os.path.join(saving_dir, f"{subclass_safe}_approx_kde.tif")
    tifffile.imwrite(out_path, density.astype(np.float32))
    print(f"✅ Saved: {out_path}")
