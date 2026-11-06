from datasets import load_from_disk
from project_paths import RAW_DATASET_PATH, PROCESSED_DATASET_PATH
from label_config import LABEL_LIST

# Load the original dataset
dataset = load_from_disk(str(RAW_DATASET_PATH))

# Get the original NER label names
original_label_names = dataset["train"].features["ner_tags"].feature.names

# Use the centralized project label list
new_label_names = LABEL_LIST

# Create a mapping from original labels to new labels
label_mapping = {}

for label_id, label_name in enumerate(original_label_names):

    if label_name in new_label_names:
        label_mapping[label_id] = new_label_names.index(label_name)

    else:
        label_mapping[label_id] = new_label_names.index("O")


# Transform the original NER labels
def transform_labels(example):

    new_ner_tags = [
        label_mapping[tag_id]
        for tag_id in example["ner_tags"]
    ]

    return {
        "ner_tags_3class": new_ner_tags
    }


# Apply the transformation to all dataset splits
processed_dataset = dataset.map(transform_labels)

# Save the processed dataset
processed_dataset.save_to_disk(str(PROCESSED_DATASET_PATH))

# Display the processed dataset
print(processed_dataset)

# Display the new label mapping
print("\nNew NER Label Mapping:\n")

for label_id, label_name in enumerate(new_label_names):
    print(f"ID: {label_id}  |  Label: {label_name}")