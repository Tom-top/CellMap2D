# Script to download STPT datasets from the Allen Brain Atlas
# using metadata provided in an Excel file. Handles selection by index,
# list of indices, or downloads all if unspecified.

import os
import multiprocessing

import pandas as pd

from src.utils import load_yaml_config, download_dataset

# === Load configuration ===
config = load_yaml_config("config/config.yml")

working_dir = config.get("working_dir")
resolution_level = config.get("resolution_level", 2)
sheet_name = config.get("sheet_name", "Test Set")
metadata_file = config.get("metadata_file", "MapMySections_EntrantData.xlsx")
bucket_name = config.get("bucket_name", "allen-genetic-tools")
dataset_selection = config.get("dataset_selection", "all")

saving_dir = os.path.join(working_dir, "data")
metadata_path = os.path.join(working_dir, metadata_file)
n_workers = max(1, multiprocessing.cpu_count() - 1)

# === Load metadata file ===
metadata = pd.read_excel(metadata_path, sheet_name=sheet_name)
all_links = metadata["STPT Data File Path"]
max_index = len(all_links) - 1

# === Dataset selection logic ===
if isinstance(dataset_selection, int):
    if dataset_selection < 0 or dataset_selection > max_index:
        raise IndexError(f"Index {dataset_selection} is out of bounds (max: {max_index})")
    dataset_links = [all_links.iloc[dataset_selection]]
    print(f"🔢 Downloading dataset at index {dataset_selection}")

elif isinstance(dataset_selection, list):
    invalid = [i for i in dataset_selection if i < 0 or i > max_index]
    if invalid:
        raise IndexError(f"Indices out of bounds: {invalid} (max: {max_index})")
    dataset_links = [all_links.iloc[i] for i in dataset_selection]
    print(f"🔢 Downloading datasets at indices: {dataset_selection}")

else:
    dataset_links = all_links
    print(f"🔢 Downloading all {len(dataset_links)} datasets")

# === Download files from bucket ===
for dataset_link in dataset_links:
    download_dataset(
        dataset_link=dataset_link,
        saving_dir=saving_dir,
        resolution_level=resolution_level,
        bucket_name=bucket_name,
        n_workers=n_workers
    )
