import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

os.environ["QT_API"] = "none"

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np


def label_to_rgb(slice_annotation, seed=42):
    """
    Map each unique label in the annotation slice to a random RGB color.
    Label 0 is always black.

    Parameters:
        slice_annotation (np.ndarray): 2D array with label values.
        seed (int): Random seed for reproducibility.

    Returns:
        rgb_image (np.ndarray): RGB image (H, W, 3).
    """
    unique_labels = np.unique(slice_annotation)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (0)

    # Generate random colors
    rng = np.random.default_rng(seed)
    colors = rng.integers(50, 255, size=(len(unique_labels), 3), dtype=np.uint8)

    # Create mapping dict
    label_to_color = {label: color for label, color in zip(unique_labels, colors)}
    label_to_color[0] = np.array([0, 0, 0], dtype=np.uint8)  # Ensure background is black

    # Create RGB image
    h, w = slice_annotation.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        mask = slice_annotation == label
        rgb_image[mask] = color

    return rgb_image


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def list_s3_keys(bucket, prefix):
    """
    List all object keys under a given prefix in an S3 bucket.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys = []
    for page in pages:
        if "Contents" in page:
            keys.extend(obj["Key"] for obj in page["Contents"])
    return keys


def download_file_from_s3(s3, bucket, key, local_base, prefix):
    """
    Download a single file from S3, skipping if it already exists locally.
    """
    rel_path = key[len(prefix):]
    local_path = os.path.join(local_base, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        return key

    try:
        s3.download_file(bucket, key, local_path)
        return key
    except Exception as e:
        return f"ERROR: {key} - {e}"


def download_dataset(dataset_link, saving_dir, resolution_level, bucket_name, n_workers):
    """
    Given a dataset S3 link, download the corresponding Zarr data from S3.
    """
    print("\n")
    dataset = dataset_link.split("/")[4]

    if "ome_zarr_conversion" not in dataset_link:
        s3_prefix = f"tissuecyte/{dataset}/ome-zarr/{resolution_level}/"
    else:
        s3_prefix = f"tissuecyte/{dataset}/ome_zarr_conversion/{dataset}.zarr/{resolution_level}/"

    local_dir = os.path.join(saving_dir, f"{dataset}.zarr", str(resolution_level))

    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"✅ Dataset for experiment '{dataset}' already exists.")
        return

    print(f"📥 Downloading dataset for experiment: {dataset}")
    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    keys = list_s3_keys(bucket_name, s3_prefix)

    print(f"📦 Found {len(keys)} files to download.")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(download_file_from_s3, s3, bucket_name, key, local_dir, s3_prefix)
            for key in keys
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {dataset}"):
            pass

    print("✅ Download complete.")


def generate_grid_lines(width, height, n_lines=50, n_points_per_line=1000):
    """
    Generates a dense grid of horizontal and vertical lines across a 2D image plane.
    Useful for overlaying reference gridlines on transformed slices.

    Parameters:
        width (int): Width of the image (in pixels).
        height (int): Height of the image (in pixels).
        n_lines (int): Number of lines to draw per direction (horizontal & vertical).
        n_points_per_line (int): Number of interpolated points per line.

    Returns:
        ndarray: (N x 2) array of (x, y) grid points for all lines.
    """
    # === Define line positions ===
    xs = np.linspace(0, width - 1, n_lines)
    ys = np.linspace(0, height - 1, n_lines)

    lines = []

    # ➖ Horizontal lines (vary x, fixed y)
    for y in ys:
        x_line = np.linspace(0, width - 1, n_points_per_line)
        y_line = np.full_like(x_line, y)
        lines.append(np.stack([x_line, y_line], axis=1))

    # ➕ Vertical lines (vary y, fixed x)
    for x in xs:
        y_line = np.linspace(0, height - 1, n_points_per_line)
        x_line = np.full_like(y_line, x)
        lines.append(np.stack([x_line, y_line], axis=1))

    # 🧩 Combine all lines into a single (N x 2) array
    return np.concatenate(lines, axis=0)
