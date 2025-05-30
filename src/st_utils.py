import numpy as np

def prepare_st_data(cell_coordinates, reference, mirror, cell_color):
    """
    Prepares spatial transcriptomics (ST) cell coordinates for projection and visualization.

    Parameters
    ----------
    cell_coordinates : pd.DataFrame
        DataFrame containing 3D cell coordinates with columns 'x', 'y', and 'z'.
        May also contain a color column (e.g., 'cluster_color', 'gene_color').

    reference : np.ndarray
        3D reference image used for mirroring and coordinate flipping (shape: [Z, X, Y]).

    mirror : bool
        Whether to mirror the data across the midline (Z-axis) to simulate both hemispheres.

    cell_color : str
        Name of the column in `cell_coordinates` used to assign colors to cells.
        If not ending with "color", all cells will be assigned red.

    Returns
    -------
    dict
        Dictionary with:
        - "coordinates": ndarray of transformed (Z, X, Y) coordinates, possibly mirrored.
        - "color_handler": function(slice_mask, in_bounds_mask) -> array or string
          that returns per-cell color values for a given selection mask.
    """

    # Reorder coordinates to match reference axis order: Z, X, Y
    coords = np.array([cell_coordinates["z"], cell_coordinates["x"], cell_coordinates["y"]]).T

    # Optionally mirror across the Z-axis (left-right flipping for brain hemispheres)
    if mirror:
        mirrored = coords.copy()
        mirrored[:, 0] = reference.shape[0] - mirrored[:, 0]  # Flip Z-axis
        coords = np.concatenate([coords, mirrored], axis=0)

    # Flip X and Y axes to match image orientation (e.g., top-left origin)
    coords[:, 1] = reference.shape[1] - coords[:, 1]  # Flip X
    coords[:, 2] = reference.shape[2] - coords[:, 2]  # Flip Y

    def get_colors(slice_mask, in_bounds_mask):
        """
        Returns cell colors for the selected 2D slice.

        Parameters
        ----------
        slice_mask : np.ndarray
            Boolean array indicating which cells are in the 3D mask (projected slab).
        in_bounds_mask : np.ndarray
            Boolean array indicating which of those cells fall inside the 2D slice bounds.

        Returns
        -------
        np.ndarray or str
            Per-cell color values (if available) or a constant color string.
        """
        if cell_color.endswith("color"):
            color_vals = cell_coordinates[f"{cell_color}"]
            if mirror:
                # Duplicate colors for mirrored cells
                color_vals = np.concatenate([color_vals, color_vals], axis=0)
            return color_vals[slice_mask][in_bounds_mask]
        else:
            return "red"

    return {"coordinates": coords, "color_handler": get_colors}
