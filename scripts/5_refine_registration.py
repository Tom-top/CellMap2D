# Script to execute the full registration and visualization pipeline.
# Loads configuration parameters from a YAML file and runs registration across all selected slices.

from src.utils import load_yaml_config
from src.registration import run_registration


# Load config file
config_path = "config/config.yml"
config = load_yaml_config(config_path)

# Run the full registration and visualization pipeline
run_registration(config)