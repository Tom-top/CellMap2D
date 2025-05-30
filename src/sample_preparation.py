import os

# Disable Qt dependency warnings and image size limits
os.environ["QT_API"] = "none"
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import zarr
import numpy as np
import pandas as pd
import cv2
from natsort import natsorted


def prepare_samples_for_registration(
    working_dir,
    samples_to_process=None,
    resolution_level=2,
    downsampling_factor=1.0,
    rotation_angle=0,
    default_auto_channel=0,
):
    """
    Prepares raw section images from selected samples for downstream registration.

    This includes:
    - Selecting the correct imaging channel (from metadata).
    - Extracting image slices from Zarr pyramids.
    - Downsampling and rotating slices.
    - Saving results to disk as JPEGs.

    Parameters:
        working_dir (str): Path to the working directory with 'data/' and 'sample_metadata.xlsx'.
        samples_to_process (list or None): List of sample folder names to process. If None, processes all.
        resolution_level (int): Resolution level to extract from the Zarr pyramid.
        downsampling_factor (float): Downsampling factor to apply (1.0 = no downsampling).
        rotation_angle (int): Rotation angle in 90° steps (0–3).
        default_auto_channel (int): Default imaging channel to use if metadata is unavailable.
    """
    data_dir = os.path.join(working_dir, "data")

    if not os.path.exists(data_dir):
        raise RuntimeError(f"Data directory not found: {data_dir}")

    all_samples = natsorted(os.listdir(data_dir))
    sample_folders = [
        os.path.join(data_dir, f)
        for f in all_samples
        if (samples_to_process is None or f in samples_to_process)
    ]

    if not sample_folders:
        raise RuntimeError("No matching sample folders found — nothing to process.")

    section_metadata_file = os.path.join(working_dir, "sample_metadata.xlsx")
    if os.path.exists(section_metadata_file):
        section_metadata = pd.read_excel(section_metadata_file)
        print(f"📖 Loaded metadata from: {section_metadata_file}")
    else:
        section_metadata = None
        print(f"⚠️ Metadata file not found — defaulting to channel {default_auto_channel} for all samples.")

    for sample_folder in sample_folders:
        sample_id = os.path.basename(sample_folder).split(".")[0]

        if section_metadata is not None:
            try:
                channel = int(section_metadata.loc[
                    section_metadata["dataset_id"] == int(sample_id),
                    "segmentation_channel"
                ].values[0])
                channel = 1 if channel == 0 else 0
            except (IndexError, KeyError, ValueError):
                print(f"⚠️ Channel metadata missing or invalid for sample {sample_id}, using default channel {default_auto_channel}.")
                channel = default_auto_channel
        else:
            channel = default_auto_channel

        extract_and_save_slices(
            sample_folder=sample_folder,
            resolution_level=resolution_level,
            channel=channel,
            downsampling_factor=downsampling_factor,
            rotation_angle=rotation_angle
        )


def extract_and_save_slices(
    sample_folder,
    resolution_level,
    channel,
    downsampling_factor,
    rotation_angle,
):
    """
    Extracts image slices from a Zarr file and saves them as processed JPEGs.

    Parameters:
        sample_folder (str): Path to the sample's Zarr folder.
        resolution_level (int): Resolution level to extract.
        channel (int): Channel index to use (e.g., autofluorescence).
        downsampling_factor (float): Downsampling factor.
        rotation_angle (int): Number of 90° rotations to apply.
    """
    if not sample_folder.endswith(".zarr"):
        raise RuntimeError(f"Unsupported sample format: {sample_folder}")

    sample_name = os.path.splitext(os.path.basename(sample_folder))[0]
    print(f"📦 Processing Zarr group: {sample_name} | Channel: {channel}")

    resolution_folder = os.path.join(sample_folder, str(resolution_level))
    arr = zarr.open(resolution_folder, mode="r")

    if arr.ndim != 4:  # Expecting [C, Z, Y, X]
        raise ValueError(f"Unexpected Zarr array shape: {arr.shape} in {sample_folder}")

    for z_index in range(arr.shape[1]):
        image_folder = os.path.join(resolution_folder, f"slice_{z_index}")
        registration_folder = os.path.join(image_folder, "registration")
        os.makedirs(registration_folder, exist_ok=True)

        filename = f"slide_ds_s{str(z_index).zfill(3)}.jpg"
        dst = os.path.join(registration_folder, filename)

        if os.path.exists(dst):
            print(f"⏭️  Skipping: {dst} already exists.")
            continue

        image = arr[channel, z_index, :, :]
        processed = downsample_and_rotate_image(image, downsampling_factor, rotation_angle)
        Image.fromarray(processed).save(dst)
        print(f"✅ Saved: {dst}")


def downsample_and_rotate_image(image, downsampling_factor, rotation_angle):
    """
    Applies downsampling and rotation to a 2D grayscale image.

    Parameters:
        image (np.ndarray): 2D grayscale image.
        downsampling_factor (float): Downsampling factor (>1 = reduce).
        rotation_angle (int): Number of 90° rotations (clockwise).

    Returns:
        np.ndarray: Processed image as uint8.
    """
    # Downsample if needed
    if downsampling_factor > 1.0:
        new_size = (
            int(image.shape[1] / downsampling_factor),
            int(image.shape[0] / downsampling_factor)
        )
        print(f"📉 Downsampling from {image.shape[::-1]} to {new_size}")
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # Rotate if requested
    if rotation_angle:
        image = np.rot90(image, rotation_angle)

    # Convert to 8-bit
    if image.dtype != np.uint8:
        max_val = image.max()
        if max_val > 0:
            image = np.clip((image / max_val) * 255, 0, 255).astype(np.uint8)
        else:
            print("⚠️  Max value is zero — image set to black.")
            image = np.zeros_like(image, dtype=np.uint8)

    return image
