import os

from src.utils import load_yaml_config
from src.process_densities import process_sample

# --- Load configuration ---
config = load_yaml_config("config/config.yml")

working_dir = config.get("working_dir", "")
data_dir = os.path.join(working_dir, "data")
samples_to_process = config.get("samples_to_process", [])
subclass_kde_dir = os.path.join(data_dir, "subclass_kde")
subclass_threshold = config.get("subclass_threshold", 0.05)

# --- Paths ---
data_dir = os.path.join(working_dir, "data")
os.makedirs(subclass_kde_dir, exist_ok=True)

def main():
    for sample_id in samples_to_process:
        sample_id = sample_id.split(".")[0]
        process_sample(
            sample_id,
            data_dir=data_dir,
            subclass_kde_dir=subclass_kde_dir,
            subclass_threshold=subclass_threshold,
        )


if __name__ == "__main__":
    main()
