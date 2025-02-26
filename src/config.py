import os

# Base directory for the project (modify as needed)
BASE_DIR = os.path.join(".")

# Define paths relative to BASE_DIR
PATH_TO_MASKS = os.path.join(BASE_DIR, "Tiles_mask_tif")
PATH_TO_DATASET = os.path.join(BASE_DIR, "Tiles_tif_no_background")
PATH_TO_DATASET_CSV = os.path.join(BASE_DIR, "data.csv")
PATH_TO_MODELS = os.path.join(BASE_DIR, "models")
PATH_TO_OUTPUT = os.path.join(BASE_DIR, "Cam-explainability", "Resnet18v")

# Ensure directories exist
for path in [PATH_TO_MASKS, PATH_TO_DATASET, PATH_TO_MODELS, PATH_TO_OUTPUT]:
    os.makedirs(path, exist_ok=True)

# Log paths
print(f"Dataset Path: {PATH_TO_DATASET}")
print(f"Models Path: {PATH_TO_MODELS}")
print(f"Output Path: {PATH_TO_OUTPUT}")

# Define class names and mappings
CLASS_NAMES = ['healthy', 'affected']
ID_TO_NAME = {idx: name for idx, name in enumerate(CLASS_NAMES)}
