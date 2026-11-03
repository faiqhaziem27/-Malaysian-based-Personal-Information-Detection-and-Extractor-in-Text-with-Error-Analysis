from datasets import load_dataset

# Download CoNLL-2003 dataset
dataset = load_dataset(
    "tomaarsen/conll2003",
    "default"
)

# Save the downloaded dataset into our project
dataset.save_to_disk("data/raw/conll2003")

# Display dataset information
print(dataset)

# Display the number of samples in each split
for split_name, split_data in dataset.items():
    print(f"{split_name}: {len(split_data)} samples")
