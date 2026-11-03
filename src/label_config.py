# Define the NER labels used in our project
LABEL_LIST = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC"
]

# Get the total number of labels
NUM_LABELS = len(LABEL_LIST)

# Create label-to-ID mapping
LABEL2ID = {
    label: index
    for index, label in enumerate(LABEL_LIST)
}

# Create ID-to-label mapping
ID2LABEL = {
    index: label
    for index, label in enumerate(LABEL_LIST)
}


# Display label configuration
if __name__ == "__main__":

    print("Project NER Label Configuration:\n")
    print(f"Total Labels: {NUM_LABELS}\n")

    for label_id, label_name in ID2LABEL.items():
        print(f"ID: {label_id} | Label: {label_name}")
