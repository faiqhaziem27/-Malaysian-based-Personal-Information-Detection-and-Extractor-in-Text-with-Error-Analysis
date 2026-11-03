from datasets import load_from_disk
from collections import Counter

# Load original and processed datasets
original_dataset = load_from_disk("data/raw/conll2003")
processed_dataset = load_from_disk("data/processed/conll2003_3class")

# Get original and new label names
original_labels = original_dataset["train"].features["ner_tags"].feature.names

new_labels = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC"
]

# Count labels in the processed training dataset
label_counter = Counter()

for sample in processed_dataset["train"]:

    for tag_id in sample["ner_tags_3class"]:

        label_name = new_labels[tag_id]
        label_counter[label_name] += 1

# Display processed label distribution
print("\nProcessed Training Label Distribution:\n")

for label_name, count in label_counter.most_common():

    print(f"{label_name:<10} : {count}")

# Check dataset sizes
print("\nDataset Size Verification:\n")

for split in ["train", "validation", "test"]:

    original_size = len(original_dataset[split])
    processed_size = len(processed_dataset[split])

    print(
        f"{split:<12} | "
        f"Original: {original_size} | "
        f"Processed: {processed_size}"
    )

    assert original_size == processed_size, (
        f"Dataset size mismatch in {split}"
    )

print("\nDataset validation completed successfully!")


# Compare original and processed labels
original_sample = original_dataset["train"][0]
processed_sample = processed_dataset["train"][0]

print("\nToken-level Label Comparison:\n")

for token, original_id, processed_id in zip(
    original_sample["tokens"],
    original_sample["ner_tags"],
    processed_sample["ner_tags_3class"]
):

    original_label = original_labels[original_id]
    processed_label = new_labels[processed_id]

    print(
        f"{token:<15} "
        f"{original_label:<10} "
        f"{processed_label}"
    )
