# Script to run DeepSlice slice orientation prediction on a batch of samples.
# Loads configuration from a YAML file and invokes the DeepSlice prediction routine.

from src.utils import load_yaml_config
from src.deepslice_runner import run_deepslice_prediction

# Load pipeline configuration from YAML file
config = load_yaml_config("config/config.yml")

# Run DeepSlice orientation prediction pipeline
run_deepslice_prediction(config)
