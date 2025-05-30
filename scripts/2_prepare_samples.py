# Script to prepare raw slice images for registration.
# Loads parameters from config.yml and extracts downsampled, rotated, and channel-selected images
# for each dataset. Delegates core logic to sample_preparation.py.

from src.utils import load_yaml_config
from src.sample_preparation import prepare_samples_for_registration


# === Load configuration ===
config = load_yaml_config("config/config.yml")

# === Extract parameters from config ===
working_dir = config.get("working_dir")
samples_to_process = config.get("samples_to_process")  # Can be None for processing all
resolution_level = config.get("resolution_level", 2)    # Zarr resolution level
downsampling_factor = config.get("downsampling_factor", 1.0)  # For slice resolution reduction
rotation_angle = config.get("rotation_angle", 0)        # Degrees (must be multiple of 90)
default_auto_channel = config.get("default_auto_channel")  # Channel to extract for registration

# === Run preparation pipeline ===
prepare_samples_for_registration(
    working_dir=working_dir,
    samples_to_process=samples_to_process,
    resolution_level=resolution_level,
    downsampling_factor=downsampling_factor,
    rotation_angle=rotation_angle,
    default_auto_channel=default_auto_channel,
)
