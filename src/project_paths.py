from pathlib import Path

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the main data directory
DATA_DIR = PROJECT_ROOT / "data"

# Define the raw dataset directory
RAW_DATASET_PATH = DATA_DIR / "raw" / "conll2003"

# Define the processed dataset directory
PROCESSED_DATASET_PATH = DATA_DIR / "processed" / "conll2003_3class"

# Define the Malaysian raw dataset file path
MALAYSIA_RAW_DATA_PATH = DATA_DIR / "malaysia" / "malaysia_ner_samples.json"

# Define the Malaysian processed dataset directory
MALAYSIA_PROCESSED_DATASET_PATH = DATA_DIR / "processed" / "malaysia_3class"

# Define required dataset splits
REQUIRED_SPLITS = (
    "train",
    "validation",
    "test"
)
