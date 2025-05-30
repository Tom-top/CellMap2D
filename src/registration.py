import os
import json

from natsort import natsorted
import numpy as np
import tifffile
import pandas as pd
import ants
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.utils import (
    label_to_rgb,
    generate_grid_lines,
)

from src.slice_extraction import extract_slice, project_points
from src.st_utils import prepare_st_data


def save_current_figure(output_dir, filename):
    """
    Saves the current Matplotlib figure to disk at high resolution and closes it.

    Args:
        output_dir (str): Directory where the figure should be saved.
        filename (str): Name of the file to save (e.g., 'warped_cells.png').

    Notes:
        - Ensures output directory exists (creates it if needed).
        - Uses dpi=300 for publication-quality output.
        - Closes the figure to free up memory.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    plt.savefig(os.path.join(output_dir, filename), dpi=300)  # Save figure with high resolution
    plt.close()  # Close figure to avoid memory leaks in long loops


def run_registration(config: dict):
    """
    Main registration function.
    Loads reference and annotation volumes, processes each sample and its slices,
    registers them to the atlas, projects segmented cells into 3D space, and saves outputs.
    """
    # === Load core paths and metadata ===
    working_dir = config["working_dir"]
    data_dir = os.path.join(working_dir, "data")

    # 🧾 Sample selection
    if config["samples_to_process"] is not None:
        sample_folders = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f in config["samples_to_process"]
        ] if os.path.exists(data_dir) else []
    else:
        sample_folders = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
        ] if os.path.exists(data_dir) else []

    # === Load atlas reference images ===
    reference_path = os.path.join("resources", f"atlas{os.sep}aba_{config['reference']}_mouse.tif")
    if not os.path.exists(reference_path):
        raise RuntimeError("❌ Reference image not found. Please download it following this "
                           "link: https://www.dropbox.com/scl/fi/j77tw7j1ftoax2h75lbqf/atlas.zip?rl"
                           "key=vvigfewjpk8apojwgor9riffp&st=rpwb3vtn&dl=0 and place it in resources.")
    reference = tifffile.imread(reference_path)
    print("📘 Loaded reference image")

    annotation_path = os.path.join("resources", f"atlas{os.sep}aba_annotation_mouse_16b.tif")
    if not os.path.exists(annotation_path):
        raise RuntimeError("❌ Annotation image not found. Please download and place it in resources.")
    annotation = tifffile.imread(annotation_path)
    print("📙 Loaded annotation image")

    # === Load transcriptomics reference data ===
    cell_coordinates = pd.read_parquet(os.path.join("resources", "atlas", "aba_combined_data.parquet"))

    # === Core config parameters ===
    slice_thickness = config.get("slice_thickness", 10)
    slides_to_process = config.get("slices_to_process", None)
    cell_color = config.get("cell_color", "cluster_color")
    default_non_linear = config.get("default_non_linear", False)
    mirror = config.get("mirror", True)
    downsampling_factor = config.get("downsampling_factor", 5)
    raw_data_max = config.get("raw_data_max", 255)
    raw_data_cmap = config.get("raw_data_cmap", "bone")
    cell_size = config.get("cell_size", 0.5)

    # === Loop through each sample folder ===
    for sample_folder in sample_folders:
        resolution_folder = os.path.join(sample_folder, str(config["resolution_level"]))
        with open(os.path.join(resolution_folder, "deepslice_results.json"), "r") as f:
            deepslice_data = json.load(f)

        # === Determine slices to process ===
        if not slides_to_process or slides_to_process is None:
            image_list = [
                os.path.join(resolution_folder, f)
                for f in natsorted(os.listdir(resolution_folder))
                if f.startswith("slice")
            ]
        else:
            image_list = [os.path.join(resolution_folder, f"slice_{s}") for s in slides_to_process]

        all_segmented_coords = []

        for image in image_list:
            print(f"\n🧠 Processing slice: {image}")
            coords_3d_file = os.path.join(image, "coords3d.npy")

            if os.path.exists(coords_3d_file):
                coords_3d = np.load(coords_3d_file, allow_pickle=True)
            else:
                coords_3d = process_single_slice(
                    image=image,
                    reference=reference,
                    annotation=annotation,
                    deepslice_data=deepslice_data,
                    cell_coordinates=cell_coordinates,
                    slice_thickness=slice_thickness,
                    default_non_linear=default_non_linear,
                    mirror=mirror,
                    cell_color=cell_color,
                    downsampling_factor=downsampling_factor,
                    raw_data_max=raw_data_max,
                    raw_data_cmap=raw_data_cmap,
                    cell_size=cell_size,
                )
                np.save(coords_3d_file, coords_3d)

            all_segmented_coords.append(coords_3d)

        # === Combine and filter all slice coordinates ===
        all_segmented_coords = [i for i in all_segmented_coords if i.size > 1]
        if all_segmented_coords:
            all_segmented_coords = np.vstack(all_segmented_coords)
            seg_vox = np.round(all_segmented_coords).astype(int)
        else:
            seg_vox = np.array([])
            print("⚠️ No valid cells segmented.")

        # === Save coordinates ===
        np.savetxt(
            os.path.join(sample_folder, "all_segmented_cells_3d.csv"),
            all_segmented_coords,
            delimiter=",",
            header="x,y,z",
            comments=""
        )

        # === Build and save 3D label volume ===
        volume_shape = reference.shape
        label_volume = np.zeros(volume_shape, dtype=np.uint16)
        skipped = 0

        for i, (z, y, x) in enumerate(seg_vox):
            if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
                label_volume[z, y, x] = i + 1
            else:
                skipped += 1

        print(f"✅ Labeled {len(seg_vox) - skipped} cells; skipped {skipped} (out of bounds)")

        tifffile.imwrite(
            os.path.join(sample_folder, "segmented_cells_volume.tif"),
            label_volume,
            imagej=True
        )

        print("✅ All slices processed.\n")


def process_single_slice(
    image,
    reference,
    annotation,
    deepslice_data,
    cell_coordinates,
    slice_thickness,
    default_non_linear,
    mirror,
    cell_color,
    downsampling_factor,
    raw_data_max,
    raw_data_cmap,
    cell_size,
):
    """
    Processes a single histological slice:
    - Extracts corresponding atlas slice using DeepSlice anchoring
    - Registers reference to histology (Affine + SyN)
    - Projects ST data and segmented cells into the slice
    - Saves visualization outputs and returns 3D segmented coordinates
    """

    # === Slice info ===
    slide_index = int(image.split("_")[-1])
    slide_info = deepslice_data["slices"][slide_index]
    anchoring_matrix = np.array(slide_info["anchoring"]).reshape(3, 3)
    height, width = slide_info["height"], slide_info["width"]

    # === Load segmented centroids if available ===
    segmentation_folder = os.path.join(image, "segmentation")
    seg_path = os.path.join(segmentation_folder, "cells_centroids.npy")
    if os.path.exists(seg_path):
        segmented_cell_coordinates = np.load(seg_path)
    else:
        segmented_cell_coordinates = np.array([])

    # === Prepare output folders ===
    registration_folder = os.path.join(image, "registration")
    os.makedirs(registration_folder, exist_ok=True)

    figures_directory = os.path.join(registration_folder, "figures")
    os.makedirs(figures_directory, exist_ok=True)

    # === Extract oblique slice from 3D reference ===
    slice_reference, slice_annotation, slice_mask_3d = extract_slice(
        reference, annotation, anchoring_matrix, height, width,
        thickness_voxels=slice_thickness
    )

    tifffile.imwrite(os.path.join(registration_folder, "slice_reference.tif"), slice_reference)
    tifffile.imwrite(os.path.join(registration_folder, "slice_annotation.tif"), slice_annotation)

    # 🔁 Also save the annotation as RGB
    annotation_rgb = label_to_rgb(slice_annotation)
    tifffile.imwrite(os.path.join(registration_folder, "slice_annotation_rgb.tif"), annotation_rgb)

    # === Load raw histological image ===
    fixed_img_path = os.path.join(image, "registration", slide_info["filename"])
    fixed_img = ants.image_read(fixed_img_path)
    if fixed_img.components > 1:
        fixed_img = ants.from_numpy(fixed_img.numpy().mean(axis=2))  # Convert RGB to grayscale

    moving_img = ants.image_read(os.path.join(registration_folder, "slice_reference.tif"))

    # === Downsample images for registration ===
    width_ds, height_ds = int(width * 1), int(height * 1)  # Keep scale consistent
    fixed_ds = ants.resample_image(fixed_img, (width_ds, height_ds), use_voxels=True)
    moving_ds = ants.resample_image(moving_img, (width_ds, height_ds), use_voxels=True)

    # 🖼 Plot target histology slice
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(np.flip(np.rot90(fixed_ds.numpy(), 3), 1), cmap=raw_data_cmap, vmin=0, vmax=raw_data_max)
    ax.set_title("Target slice")
    ax.axis("off")
    ax.set_xlim(0, fixed_ds.shape[0])
    ax.set_ylim(0, fixed_ds.shape[1])
    ax.invert_yaxis()
    plt.tight_layout()
    save_current_figure(figures_directory, "target_slice.png")

    # === Register atlas reference slice to histology ===
    print("🔗 Registering reference to histology...")
    ref_to_raw_directory = os.path.join(registration_folder, "ref_to_auto")
    os.makedirs(ref_to_raw_directory, exist_ok=True)

    syn_warp, warped_field = register_images(fixed_ds, moving_ds, ref_to_raw_directory, use_default_syn=default_non_linear)

    if syn_warp is None or warped_field is None:
        print("❌ Registration failed — skipping.")
        return np.array([])

    # === Warp the annotation slice to the histological space ===
    annotation_img = ants.image_read(os.path.join(registration_folder, "slice_annotation.tif"))
    annotation_ds = ants.resample_image(annotation_img, (width_ds, height_ds), use_voxels=True)
    transform_images(fixed_ds, annotation_ds, ref_to_raw_directory, output_name="annotation")

    warped_annot = tifffile.imread(os.path.join(ref_to_raw_directory, "transformed_annotation.tif"))
    warped_annot_rgb = label_to_rgb(warped_annot)
    tifffile.imwrite(os.path.join(ref_to_raw_directory, "transformed_annotation_rgb.tif"), warped_annot_rgb)

    # === Load and project spatial transcriptomics data ===
    st_data = prepare_st_data(cell_coordinates, reference, mirror, cell_color)

    st_coords_in_slice, slice_mask, in_bounds_mask = project_points(
        st_data["coordinates"],
        anchoring_matrix,
        width,
        height,
        thickness_voxels=slice_thickness
    )

    cell_colors_in_slice = st_data["color_handler"](slice_mask, in_bounds_mask)
    grid_in_slice = generate_grid_lines(width=height, height=width)

    # === Warp and plot spatial transcriptomics + segmentation overlay ===
    warped_segmented_arr = warp_and_plot_results(
        moving_img, fixed_ds, moving_ds, syn_warp, warped_field,
        st_coords_in_slice, grid_in_slice, figures_directory,
        transform_dir=ref_to_raw_directory,
        cell_colors=cell_colors_in_slice,
        cell_color_mode="gene" if cell_color.endswith("gene") else "color",
        segmented_cell_coordinates=segmented_cell_coordinates,
        cell_size=cell_size,
        downsampling_factor=downsampling_factor,
    )

    # === Recompute segmented cell coordinates in 3D atlas space ===
    if warped_segmented_arr.size > 0:
        o, u, v = anchoring_matrix
        u_hat = u / np.linalg.norm(u)
        v_proj = v - np.dot(v, u_hat) * u_hat
        v_hat = v_proj / np.linalg.norm(v_proj)

        segmented_3d_coords = (
            o[None, :]
            + warped_segmented_arr[:, 0:1] * (u_hat / width * np.linalg.norm(u))
            + warped_segmented_arr[:, 1:2] * (v_hat / height * np.linalg.norm(v_proj))
        )
    else:
        segmented_3d_coords = np.array([])

    return segmented_3d_coords


def register_images(fixed, moving, output_dir, use_default_syn):
    """
    Registers a moving image (atlas slice) to a fixed image (raw histology).
    Performs Affine followed by SyN (non-linear) registration.

    Parameters:
        fixed (ANTsImage): Target image (histology section).
        moving (ANTsImage): Source image (atlas slice).
        output_dir (str): Directory to save intermediate and final registration files.
        use_default_syn (bool): If True, use standard SyN registration with affine pre-alignment.

    Returns:
        Tuple: (warped_image, deformation_field_array) or (None, None) if registration fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save intermediate NIfTI files for ANTs registration
    fixed_path = os.path.join(output_dir, "fixed_image.nii")
    moving_path = os.path.join(output_dir, "moving_image.nii")
    ants.image_write(fixed, fixed_path)
    ants.image_write(moving, moving_path)

    try:
        # Step 1 — Affine registration
        affine_reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine",
            outprefix=os.path.join(output_dir, "affine_"),
            verbose=True
        )
        affine_transform_params = affine_reg["fwdtransforms"][0]
        affine_warp = affine_reg["warpedmovout"]
    except Exception as e:
        print(f"❌ Affine registration failed: {e}")
        return None, None

    try:
        # Step 2 — Non-linear SyN registration (optional Affine pre-alignment)
        if use_default_syn:
            syn_reg = ants.registration(
                fixed=fixed,
                moving=affine_warp,
                type_of_transform="SyN",
                initial_transform=affine_transform_params,
                outprefix=os.path.join(output_dir, "syn_"),
                verbose=True
            )
        else:
            syn_reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform="SyNOnly",
                initial_transform=affine_transform_params,
                grad_step=0.8,
                flow_sigma=20,
                total_sigma=10,
                syn_metric="mattes",
                outprefix=os.path.join(output_dir, "syn_"),
                verbose=True
            )

        syn_warp = syn_reg["warpedmovout"]
        deformation_field = nib.load(os.path.join(output_dir, "syn_1Warp.nii.gz")).get_fdata()
    except Exception as e:
        print(f"❌ SyN registration failed: {e}")
        return None, None

    return syn_warp, deformation_field


def transform_images(fixed, moving, output_dir, output_name=""):
    """
    Applies saved transformations (Affine + SyN) to warp a new image into the fixed space.

    Parameters:
        fixed (ANTsImage): Target space for alignment (usually histology).
        moving (ANTsImage): Image to warp (e.g., annotation).
        output_dir (str): Directory with saved transforms.
        output_name (str): Prefix used for output files.
    """
    # === Load combined transform list ===
    transform_list = [
        os.path.join(output_dir, "syn_1Warp.nii.gz"),                # Deformation field
        os.path.join(output_dir, "affine_0GenericAffine.mat")        # Affine transform
    ]

    # === Apply transformation ===
    transformed = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transform_list,
        interpolator="nearestNeighbor"
    )

    # === Save transformed image as RGB (TIFF) ===
    transformed_arr = transformed.numpy()
    transformed_arr = np.swapaxes(transformed_arr, 0, 1)  # Adjust orientation for image convention

    transformed_path = os.path.join(output_dir, f"transformed_{output_name}.tif")
    tifffile.imwrite(transformed_path, transformed_arr)


def warp_and_plot_results(
    moving_img, fixed_ds, moving_ds, syn_transform, warped_field,
    st_coords, grid_coords, figures_directory,
    transform_dir,
    cell_colors, cell_color_mode, raw_data_cmap="bone_r", cell_size=0.25,
    segmented_cell_coordinates=None, downsampling_factor=1,
):
    """
    Warps spatial transcriptomics data and segmentation results from histology into atlas space.
    Saves multiple visualization outputs (raw/wrapped grids, ST cells, deformations).

    Returns:
        np.ndarray: Warped segmented cell coordinates in atlas space.
    """

    def base_plot(img, title, cmap="bone_r", alpha=0.5):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(np.flip(np.rot90(img, 3), 1), cmap=cmap, vmin=0, vmax=np.max(img), alpha=alpha)
        ax.set_title(title)
        ax.axis("off")
        ax.set_xlim(0, img.shape[0])
        ax.set_ylim(0, img.shape[1])
        ax.invert_yaxis()
        plt.tight_layout()
        return fig, ax

    # === Compute scaling factors between full-res and downsampled space ===
    spacing_x, spacing_y = moving_img.spacing
    spacing_x_ds, spacing_y_ds = moving_ds.spacing
    scale_x = spacing_x / spacing_x_ds
    scale_y = spacing_y / spacing_y_ds

    # === Scale input coordinates ===
    st_coords_ds = np.stack([
        st_coords[:, 1] * scale_y,  # x
        st_coords[:, 0] * scale_x   # y
    ], axis=1)

    grid_coords_ds = np.stack([
        grid_coords[:, 1] * scale_y,
        grid_coords[:, 0] * scale_x
    ], axis=1)

    # === Plot: raw atlas slice (moving image) ===
    fig, ax = base_plot(moving_ds.numpy(), "Predicted slice")
    save_current_figure(figures_directory, "predicted_slice.png")

    # === Plot: raw grid overlay ===
    fig, ax = base_plot(moving_ds.numpy(), "Raw grid points")
    ax.scatter(grid_coords_ds[:, 0], grid_coords_ds[:, 1], c="red", s=0.2, alpha=0.05)
    save_current_figure(figures_directory, "raw_grid_points.png")

    # === Plot: ST cells before warping ===
    fig, ax = base_plot(moving_ds.numpy(), "Raw ST cell points")
    ax.scatter(
        st_coords_ds[:, 0], st_coords_ds[:, 1],
        c=cell_colors, s=cell_size, edgecolors='none', marker='o',
        alpha=1.0 if cell_color_mode.endswith("gene") else 0.5
    )
    save_current_figure(figures_directory, "raw_st_cells.png")

    # === Convert to physical space for transformation ===
    st_phys = pd.DataFrame({
        "x": st_coords_ds[:, 0] * spacing_x_ds,
        "y": st_coords_ds[:, 1] * spacing_y_ds,
    })
    grid_phys = pd.DataFrame({
        "x": grid_coords_ds[:, 0] * spacing_x_ds,
        "y": grid_coords_ds[:, 1] * spacing_y_ds,
    })

    try:
        # 🧬 Warp ST coordinates from histology → atlas
        warped_st = ants.apply_transforms_to_points(
            dim=2,
            points=st_phys,
            transformlist=[
                os.path.join(transform_dir, 'syn_1InverseWarp.nii.gz'),
                os.path.join(transform_dir, 'syn_0GenericAffine.mat')
            ],
            whichtoinvert=[False, True]
        )
    except ValueError:
        return np.array([])

    # Warp grid points the same way
    warped_grid = ants.apply_transforms_to_points(
        dim=2,
        points=grid_phys,
        transformlist=[
            os.path.join(transform_dir, 'syn_1InverseWarp.nii.gz'),
            os.path.join(transform_dir, 'syn_0GenericAffine.mat')
        ],
        whichtoinvert=[False, True]
    )

    warped_st_arr = np.stack([
        warped_st["x"] / spacing_x_ds,
        warped_st["y"] / spacing_y_ds
    ])

    warped_grid_arr = np.stack([
        warped_grid["x"] / spacing_x_ds,
        warped_grid["y"] / spacing_y_ds
    ], axis=1)

    # === Plot: warped grid over reference ===
    fig, ax = base_plot(syn_transform.numpy(), "Warped grid points")
    ax.scatter(warped_grid_arr[:, 0], warped_grid_arr[:, 1], c="red", s=0.2, alpha=0.05)
    save_current_figure(figures_directory, "warped_grid_points.png")

    # === Plot: warped ST cells on warped atlas ===
    fig, ax = base_plot(syn_transform.numpy(), "Warped ST cells (on ref)")
    ax.scatter(
        warped_st_arr[0], warped_st_arr[1],
        c=cell_colors, s=cell_size, edgecolors='none', marker='o',
        alpha=1.0 if cell_color_mode.endswith("gene") else 0.5
    )
    save_current_figure(figures_directory, "warped_st_cells_on_ref.png")

    # === Plot: warped ST cells over histology ===
    fig, ax = base_plot(fixed_ds.numpy(), "Warped ST cells (on raw)", raw_data_cmap, alpha=1)
    ax.scatter(
        warped_st_arr[0], warped_st_arr[1],
        c=cell_colors, s=cell_size, edgecolors='none', marker='o',
        alpha=1.0 if cell_color_mode.endswith("gene") else 0.5
    )
    save_current_figure(figures_directory, "warped_st_cells_on_raw.png")

    # === Overlay segmented cells (if available) ===
    fig, ax = base_plot(fixed_ds.numpy(), "Segmented and ST cells on raw", raw_data_cmap, alpha=1)
    ax.scatter(
        warped_st_arr[0], warped_st_arr[1],
        c=cell_colors, s=cell_size, edgecolors='none', marker='o',
        alpha=1.0 if cell_color_mode.endswith("gene") else 0.5
    )
    if segmented_cell_coordinates is not None and segmented_cell_coordinates.size > 0:
        ax.scatter(
            segmented_cell_coordinates[:, 1] / 10,
            segmented_cell_coordinates[:, 2] / 10,
            c="blue", s=1, edgecolors='none', marker='o', alpha=0.5
        )
    save_current_figure(figures_directory, "warped_st_cells_and_segmented_on_raw.png")

    # === Plot deformation fields ===
    fig, ax = base_plot(moving_ds.numpy(), "Horizontal deformation field")
    ax.imshow(np.flip(np.rot90(warped_field[:, :, 0, 0, 0], 3), 1), cmap='RdBu', vmin=-50, vmax=50, alpha=0.5)
    save_current_figure(figures_directory, "horizontal_deformation_field.png")

    fig, ax = base_plot(moving_ds.numpy(), "Vertical deformation field")
    ax.imshow(np.flip(np.rot90(warped_field[:, :, 0, 0, 1], 3), 1), cmap='RdBu', vmin=-50, vmax=50, alpha=0.5)
    save_current_figure(figures_directory, "vertical_deformation_field.png")

    # === Warp segmented cells (optional) ===
    if segmented_cell_coordinates is not None and segmented_cell_coordinates.size > 0:
        segmented_phys = pd.DataFrame({
            "x": segmented_cell_coordinates[:, 1] / downsampling_factor * spacing_x_ds,
            "y": segmented_cell_coordinates[:, 2] / downsampling_factor * spacing_y_ds,
        })

        warped_segmented = ants.apply_transforms_to_points(
            dim=2,
            points=segmented_phys,
            transformlist=[
                os.path.join(transform_dir, 'syn_1Warp.nii.gz'),
                os.path.join(transform_dir, 'syn_0GenericAffine.mat')
            ],
            whichtoinvert=[False, False]
        )

        warped_segmented_arr = np.stack([
            warped_segmented["x"] / spacing_x_ds,
            warped_segmented["y"] / spacing_y_ds
        ], axis=1)

        # Final overlay
        fig, ax = base_plot(moving_ds.numpy(), "Warped segmentation")
        ax.scatter(
            warped_segmented_arr[:, 0], warped_segmented_arr[:, 1],
            c="blue", s=1, edgecolors='none', marker='o', alpha=0.5
        )
        save_current_figure(figures_directory, "warped_segmentation.png")
    else:
        warped_segmented_arr = np.array([])

    return warped_segmented_arr
