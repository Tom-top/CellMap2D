import os
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter

from src.utils import load_yaml_config

# === Load config ===
config = load_yaml_config("config/config.yml")

working_dir = config.get("working_dir", "")
datasets = config.get("samples_to_process", [])
reference_shape = tuple(config.get("reference_shape", (456, 528, 320)))  # (x, y, z)
downsampling_factor = config.get("density_downsampling", 4)
voxel_size_mm = config.get("voxel_size_mm", 0.025)
bandwidth_mm = config.get("kde_bandwidth_mm", 0.2)

# === Derived parameters ===
lowres_shape = tuple(s // downsampling_factor for s in reference_shape)
blur_sigma = bandwidth_mm / (voxel_size_mm * downsampling_factor)

# === Process each dataset ===
for dataset in datasets:
    print(f"\n📦 Processing dataset: {dataset}")

    dataset_name = dataset.split(".")[0]
    dataset_dir = os.path.join(working_dir, "data", f"{dataset}")
    csv_path = os.path.join(dataset_dir, "all_segmented_cells_3d.csv")

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found for dataset {dataset}, skipping.")
        continue

    # Load CSV (auto-handle tab or comma separator)
    with open(csv_path) as f:
        first_line = f.readline()
    df = pd.read_csv(csv_path, sep="\t") if "\t" in first_line else pd.read_csv(csv_path)

    coords = df[["x", "y", "z"]].values

    # Downsample coordinates
    coords_ds = (coords / downsampling_factor).round().astype(int)

    # Initialize density volume
    density = np.zeros(lowres_shape, dtype=np.float32)

    # Populate grid
    valid = (
        (coords_ds[:, 0] >= 0) & (coords_ds[:, 0] < lowres_shape[0]) &
        (coords_ds[:, 1] >= 0) & (coords_ds[:, 1] < lowres_shape[1]) &
        (coords_ds[:, 2] >= 0) & (coords_ds[:, 2] < lowres_shape[2])
    )
    coords_ds = coords_ds[valid]
    np.add.at(density, (coords_ds[:, 0], coords_ds[:, 1], coords_ds[:, 2]), 1)

    # Apply Gaussian KDE blur
    density = gaussian_filter(density, sigma=blur_sigma)

    # Save inside dataset's folder
    out_path = os.path.join(dataset_dir, f"{dataset_name}_kde.tif")
    tifffile.imwrite(out_path, density.astype(np.float32))
    print(f"✅ Saved downsampled KDE to {out_path}")
