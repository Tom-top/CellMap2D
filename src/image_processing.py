import os
import time
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import zarr
import numpy as np
import tifffile
import cv2

from tqdm import tqdm
from scipy.ndimage import gaussian_laplace, distance_transform_edt, label as ndi_label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.draw import disk


def run_segmentation_on_slice(z_index, zarr_path, channel, dataset_id, save_outputs=True):
    """
    Wrapper for multiprocessing: loads zarr array and runs segmentation on a single slice.

    Parameters:
        z_index (int): Index of the slice.
        zarr_path (str): Path to the zarr resolution folder.
        channel (int): Channel index to use.
        dataset_id (str): Dataset identifier for logging.
        save_outputs (bool): Whether to save intermediate and output files to disk.

    Returns:
        str: Status message from slice segmentation.
    """
    arr = zarr.open(zarr_path, mode='r')
    return segment_single_slice(z_index, arr, channel, zarr_path, dataset_id, save_outputs)


def segment_single_slice(z_index, arr, channel, zarr_path, dataset_id, save_outputs=True):
    """
    Segments blobs in a single z-slice of a 3D image volume.

    Parameters:
        z_index (int): Slice index.
        arr (zarr array): 4D image volume [C, Z, Y, X].
        channel (int): Channel index.
        zarr_path (str): Path to dataset.
        dataset_id (str): Identifier used for logging.
        save_outputs (bool): Whether to save output TIFFs and centroid arrays.

    Returns:
        str: Status update string.
    """
    slice_dir = os.path.join(zarr_path, f"slice_{z_index}")
    segmentation_dir = os.path.join(slice_dir, "segmentation")

    centroid_path = os.path.join(segmentation_dir, "cells_centroids.npy")
    if os.path.exists(centroid_path):
        return f"[Dataset: {dataset_id}] Skipping slice {z_index}, already segmented."

    # === Load raw image ===
    img = arr[channel, z_index, :, :].astype(np.float32)
    h, w = img.shape

    if save_outputs:
        os.makedirs(segmentation_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(segmentation_dir, "raw_slice.tif"), img)

    # === Compute tissue mask ===
    threshold = threshold_otsu(img) * 0.1
    tissue_mask = remove_small_objects(img > threshold, min_size=5000)

    labeled, _ = ndi_label(tissue_mask.astype(int))
    props = regionprops(labeled)
    if not props:
        return f"❌ No tissue detected on slice {z_index}"

    # === Crop to tissue bounding box ===
    ys, xs, ye, xe = zip(*[p.bbox for p in props])
    y0, x0 = max(0, min(ys) - 20), max(0, min(xs) - 20)
    y1, x1 = min(h, max(ye) + 20), min(w, max(xe) + 20)

    img_crop = img[y0:y1, x0:x1]

    if save_outputs:
        overlay = cv2.cvtColor((tissue_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
        tifffile.imwrite(os.path.join(segmentation_dir, "tissue_mask.tif"), overlay)

    # === Segment blobs ===
    start = time.time()
    mask_crop, centroids, _, _, rejected = segment_blobs_log_watershed(
        img_crop, sigma=2, min_area=30, max_area=1000, min_circularity=0.85
    )
    elapsed = time.time() - start

    # === Reproject results to full image ===
    full_mask = np.zeros_like(img, dtype=np.uint16)
    adjusted_centroids = []
    for cid, x, y in centroids:
        x_full, y_full = x + x0, y + y0
        rr, cc = disk((y_full, x_full), radius=5, shape=full_mask.shape)
        full_mask[rr, cc] = cid
        adjusted_centroids.append((cid, x_full, y_full))

    if save_outputs:
        tifffile.imwrite(os.path.join(segmentation_dir, "cells.tif"), full_mask)
        np.save(os.path.join(segmentation_dir, "cells_centroids.npy"), np.array(adjusted_centroids))

    return (
        f"[DONE] ⏱ {elapsed:.2f} sec — "
        f"{len(adjusted_centroids)} cells, "
        f"{np.count_nonzero(rejected)} rejected on slice {z_index}."
    )


def segment_blobs_log_watershed(img_crop, sigma=2, min_area=30, max_area=1000, min_circularity=0.85):
    """
    Performs blob segmentation using Laplacian of Gaussian and watershed, with filtering.

    Parameters:
        img_crop (np.ndarray): 2D grayscale image.
        sigma (float): Sigma for LoG filter.
        min_area (int): Minimum area threshold.
        max_area (int): Maximum area threshold.
        min_circularity (float): Minimum circularity threshold.

    Returns:
        Tuple:
            - output_mask (np.ndarray)
            - centroids (np.ndarray)
            - info (list)
            - labels_ws (np.ndarray)
            - rejected_mask (np.ndarray)
    """
    img_crop = (img_crop - img_crop.min()) / (img_crop.max() + 1e-8)
    log_img = -gaussian_laplace(img_crop, sigma=sigma)
    binary = log_img > (np.median(log_img) + 5 * np.std(log_img))
    distance = distance_transform_edt(binary)
    coords = peak_local_max(distance, labels=binary, min_distance=5)

    markers = np.zeros_like(img_crop, dtype=np.int32)
    if coords.size > 0:
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labels_ws = watershed(-distance, markers, mask=binary)
    output_mask = np.zeros_like(labels_ws, dtype=np.uint16)
    rejected_mask = np.zeros_like(labels_ws, dtype=np.uint8)
    centroids = []
    info = []
    label_id = 1

    for region in regionprops(labels_ws):
        area = region.area
        perimeter = max(region.perimeter, 1e-6)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if min_area <= area <= max_area and circularity >= min_circularity:
            output_mask[region.slice][region.image] = label_id
            cy, cx = region.centroid
            centroids.append((label_id, int(cx), int(cy)))
            info.append((label_id, int(cx), int(cy), int(area), round(circularity, 2)))
            label_id += 1
        else:
            rejected_mask[region.slice][region.image] = 255

    return output_mask, np.array(centroids), info, labels_ws, rejected_mask


def segment_entire_dataset_parallel(dataset_id, channel, zarr_path, num_processes, save_outputs=True):
    """
    Runs segmentation on all slices of a dataset using multiprocessing.

    Parameters:
        dataset_id (str): Identifier.
        channel (int): Channel index to use.
        zarr_path (Path): Path to the dataset's Zarr folder.
        num_processes (int): Number of parallel processes.
        save_outputs (bool): Whether to save segmentation results.
    """
    if not zarr_path.exists():
        print(f"⚠️  Skipping: {zarr_path} does not exist.")
        return

    arr = zarr.open(str(zarr_path), mode="r")
    z_indices = np.arange(arr.shape[1])

    print(f"🧠 Processing {len(z_indices)} slices from {dataset_id} using {num_processes} processes.")
    start_time = time.time()

    process_func = partial(
        run_segmentation_on_slice,
        zarr_path=str(zarr_path),
        channel=channel,
        dataset_id=dataset_id,
        save_outputs=save_outputs
    )

    results = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(z_indices), desc=f"Dataset {dataset_id}") as pbar:
            for result in pool.imap(process_func, z_indices):
                results.append(result)
                print(result)
                pbar.update(1)

    elapsed = time.time() - start_time
    successful = sum("[DONE]" in r for r in results)
    skipped = sum("Skipping" in r for r in results)
    failed = len(results) - successful - skipped

    print(f"\n[Dataset: {dataset_id}] Completed in {elapsed:.2f} seconds")
    print(f"[Dataset: {dataset_id}] Summary: {successful} processed, {skipped} skipped, {failed} failed")
