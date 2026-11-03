from datasets import load_from_disk

def load_ner_dataset():
    """
    Load the processed CoNLL-2003 NER dataset.

    Returns:
        DatasetDict: Contains train, validation, and test splits.
    """

    dataset_path = "data/processed/conll2003_3class"

    dataset = load_from_disk(dataset_path)

    return dataset


if __name__ == "__main__":

    # Load the dataset
    dataset = load_ner_dataset()

    # Display dataset information
    print("Dataset loaded successfully!\n")
    print(dataset)

    # Display split sizes
    print("\nDataset Split Sizes:\n")

    for split in ["train", "validation", "test"]:

        print(f"{split.capitalize()}: {len(dataset[split])} samples")

    # Display one training sample
    print("\nFirst Training Sample:\n")

    sample = dataset["train"][0]

    print("Tokens:", sample["tokens"])
    print("NER Labels:", sample["ner_tags_3class"])
