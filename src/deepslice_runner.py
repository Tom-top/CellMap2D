import os
from natsort import natsorted
from PIL import Image
from DeepSlice import DSModel

Image.MAX_IMAGE_PIXELS = None  # Disable warning for large images


def run_deepslice_prediction(config):
    """
    Runs DeepSlice prediction on sample slices and saves output JSON.
    """

    # === Load species model ===
    species = config.get("species", "mouse")
    model = DSModel(species=species)

    # === Paths ===
    working_dir = config["working_dir"]
    data_dir = os.path.join(working_dir, "data")

    # === Sample selection ===
    if config["samples_to_process"] is not None:
        sample_folders = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f in config["samples_to_process"]
        ] if os.path.exists(data_dir) else []
    else:
        sample_folders = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
        ] if os.path.exists(data_dir) else []

    # === Run prediction on each sample ===
    for sample_folder in sample_folders:

        resolution_folder = os.path.join(sample_folder, str(config["resolution_level"]))

        # 📸 Collect slice images for DeepSlice
        image_list = [
            os.path.join(
                resolution_folder,
                f,
                f"registration{os.sep}slide_ds_s{f.split('_')[-1].zfill(3)}.jpg"
            )
            for f in natsorted(os.listdir(resolution_folder))
            if f.startswith("slice")
        ]

        # 🧠 Run model prediction
        if config.get("single_sections", False):
            model.predict(image_list=image_list, ensemble=False, section_numbers=False)
        else:
            model.predict(image_list=image_list, ensemble=True, section_numbers=True)

            if config.get("propagate_angles", True):
                model.propagate_angles()

            if config.get("enforce_index_order", True):
                model.enforce_index_order()

            if config.get("enforce_index_spacing", True):
                model.enforce_index_spacing(section_thickness=None)

        # 💾 Save DeepSlice prediction results
        model.save_predictions(os.path.join(resolution_folder, "deepslice_results"))
        print("✅ DeepSlice prediction complete and saved.")
