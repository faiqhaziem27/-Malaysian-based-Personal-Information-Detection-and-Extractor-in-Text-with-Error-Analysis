from datasets import load_from_disk
try:
    from project_paths import PROCESSED_DATASET_PATH, REQUIRED_SPLITS
except ModuleNotFoundError:
    from src.project_paths import PROCESSED_DATASET_PATH, REQUIRED_SPLITS

def load_ner_dataset():
    """
    Load the processed CoNLL-2003 NER dataset.

    Returns:
        DatasetDict: Contains train, validation, and test splits.
    """

    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {PROCESSED_DATASET_PATH}. "
            "Please run 'python src/prepare_dataset.py' first."
        )

    dataset = load_from_disk(str(PROCESSED_DATASET_PATH))

    return dataset


if __name__ == "__main__":

    # Load the dataset
    dataset = load_ner_dataset()

    # Display dataset information
    print("Dataset loaded successfully!\n")
    print(dataset)

    # Display split sizes
    print("\nDataset Split Sizes:\n")

    for split in REQUIRED_SPLITS:

        print(f"{split.capitalize()}: {len(dataset[split])} samples")

    # Display one training sample
    print("\nFirst Training Sample:\n")

    sample = dataset["train"][0]

    print("Tokens:", sample["tokens"])
    print("NER Labels:", sample["ner_tags_3class"])
