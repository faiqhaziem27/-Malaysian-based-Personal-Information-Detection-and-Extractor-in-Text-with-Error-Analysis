from datasets import load_dataset
from project_paths import RAW_DATASET_PATH


def main():
    # Use the centralized raw dataset path
    raw_data_path = RAW_DATASET_PATH

    # Skip download if the raw dataset already exists
    if raw_data_path.exists():
        print(f"Raw dataset already exists at: {raw_data_path}")
        print("Skipping download. Delete the folder to re-download.")
        return

    # Download CoNLL-2003 dataset
    print("Downloading CoNLL-2003 dataset...")

    dataset = load_dataset(
        "tomaarsen/conll2003",
        "default"
    )

    # Save the downloaded dataset into our project
    dataset.save_to_disk(str(raw_data_path))

    print(f"\nDataset successfully saved to: {raw_data_path}")

    # Display dataset information
    print("\nDataset Information:")
    print(dataset)

    # Display the number of samples in each split
    print("\nDataset Split Sizes:")

    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} samples")


if __name__ == "__main__":
    main()
