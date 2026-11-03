from datasets import load_from_disk
from collections import Counter

# Load the downloaded dataset
dataset = load_from_disk("data/raw/conll2003")

# Get the NER label names
label_names = dataset["train"].features["ner_tags"].feature.names

# Select the first training sample
sample = dataset["train"][0]

# Display the tokens and their corresponding NER labels
print("First Training Sample:\n")

for token, tag_id in zip(sample["tokens"], sample["ner_tags"]):
    label = label_names[tag_id]
    print(f"{token:<20} {label}")

# Display all NER labels and their corresponding IDs
print("\nNER Label Mapping:\n")

for label_id, label_name in enumerate(label_names):
    print(f"ID: {label_id}  |  Label: {label_name}")

# Create a counter for NER labels
label_counter = Counter()

# Count all NER labels in the training dataset
for sample in dataset["train"]:

    for tag_id in sample["ner_tags"]:

        label_name = label_names[tag_id]

        label_counter[label_name] += 1

# Display the label distribution
print("\nTraining NER Label Distribution:\n")

for label_name, count in label_counter.most_common():

    print(f"{label_name:<10} : {count}")


# Count complete entities and entity tokens
entity_counter = Counter()
entity_token_counter = Counter()

for sample in dataset["train"]:

    for tag_id in sample["ner_tags"]:

        label = label_names[tag_id]

        if label.startswith("B-"):
            entity_type = label[2:]
            entity_counter[entity_type] += 1
            entity_token_counter[entity_type] += 1

        elif label.startswith("I-"):
            entity_type = label[2:]
            entity_token_counter[entity_type] += 1

# Display entity count
print("\nTraining Entity Count:\n")

for entity_type, count in sorted(entity_counter.items()):
    print(f"{entity_type:<10} : {count}")

# Display entity token count
print("\nTraining Entity Token Count:\n")

for entity_type, count in sorted(entity_token_counter.items()):
    print(f"{entity_type:<10} : {count}")

# Calculate total number of entities
total_entities = sum(entity_counter.values())

# Calculate and display entity percentages
print("\nTraining Entity Distribution (%):\n")

for entity_type, count in sorted(entity_counter.items()):

    percentage = (count / total_entities) * 100

    print(
        f"{entity_type:<10} : "
        f"{percentage:.2f}%"
    )


