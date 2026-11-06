from pathlib import Path
from datasets import load_dataset


def main():
    # Define the project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Define the raw dataset storage path
    raw_data_path = project_root / "data" / "raw" / "conll2003"

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
