# Script for batch segmentation of Zarr image volumes.
# Loads metadata, selects channels, and parallelizes slice-wise segmentation across datasets.
# Outputs log info and handles missing metadata or data gracefully.

import os
import time
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import zarr
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import load_yaml_config
from src.image_processing import run_segmentation_on_slice

# === Load configuration from YAML ===
config = load_yaml_config("config/config.yml")

working_dir = config.get("working_dir", "")
resolution_level = config.get("resolution_level", 2)
samples_to_process = config.get("samples_to_process", None)
default_seg_channel = config.get("default_seg_channel")  # Corrected typo
num_processes = config.get("num_processes", max(1, cpu_count() - 20))  # Conservative default

data_dir = os.path.join(working_dir, "data")
metadata_path = os.path.join(working_dir, "sample_metadata.xlsx")

# === Load metadata (used for per-dataset channel selection) ===
if os.path.exists(metadata_path):
    section_metadata = pd.read_excel(metadata_path)
    print(f"📖 Loaded metadata from: {metadata_path}")
else:
    section_metadata = None
    print(f"⚠️ Metadata file not found — defaulting to channel 1 for all datasets.")

# === If no sample list is provided, process all Zarr folders ===
if samples_to_process is None:
    samples_to_process = [
        f.split(".")[0] for f in os.listdir(data_dir)
        if f.endswith(".zarr") and os.path.isdir(os.path.join(data_dir, f))
    ]
    print(f"📂 No sample list provided — processing all ({len(samples_to_process)}) datasets.")
else:
    samples_to_process = [
        f.split(".")[0] for f in samples_to_process
    ]

# === Iterate through each dataset and run segmentation ===
for dataset_id in samples_to_process:
    print(f"\n[Dataset: {dataset_id}] Starting segmentation.")

    # Lookup segmentation channel from metadata
    if section_metadata is not None:
        try:
            seg_channel = int(section_metadata.loc[
                section_metadata["dataset_id"] == int(dataset_id),
                "segmentation_channel"
            ].values[0])
        except (IndexError, KeyError, ValueError):
            print(f"⚠️ No channel metadata found for {dataset_id}, defaulting to channel 1.")
            seg_channel = default_seg_channel
    else:
        seg_channel = default_seg_channel

    zarr_path = Path(os.path.join(data_dir, f"{dataset_id}.zarr", str(resolution_level)))
    if not zarr_path.exists():
        print(f"⚠️ Skipping: {zarr_path} does not exist.")
        continue

    arr = zarr.open(str(zarr_path), mode="r")
    z_indices = np.arange(arr.shape[1])  # Loop through all slices

    print(f"🧠 Processing {len(z_indices)} slices using {num_processes} processes...")

    dataset_start_time = time.time()

    # Partial function: sets static arguments for each slice
    process_func = partial(
        run_segmentation_on_slice,
        zarr_path=str(zarr_path),
        channel=seg_channel,
        dataset_id=dataset_id,
        save_outputs=True,  # Save intermediate images
    )

    results = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(z_indices), desc=f"Dataset {dataset_id}") as pbar:
            for result in pool.imap(process_func, z_indices):
                results.append(result)
                print(result)
                pbar.update(1)

    # === Summary for dataset ===
    elapsed = time.time() - dataset_start_time
    successful = sum(1 for r in results if "[DONE]" in r)
    skipped = sum(1 for r in results if "Skipping" in r)
    failed = len(results) - successful - skipped

    print(f"\n[Dataset: {dataset_id}] Completed in {elapsed:.2f} seconds")
    print(f"[Dataset: {dataset_id}] Summary: {successful} processed, {skipped} skipped, {failed} failed")
