from tokenize_dataset import load_tokenized_dataset, IGNORE_LABEL_ID
from load_processed_dataset import load_ner_dataset
from project_paths import REQUIRED_SPLITS

# Load word-level and tokenized datasets
word_dataset = load_ner_dataset()
tokenized_dataset, tokenizer = load_tokenized_dataset()

print("\nTokenization Validation:\n")

for split in REQUIRED_SPLITS:

    word_split = word_dataset[split]
    token_split = tokenized_dataset[split]

    assert len(word_split) == len(token_split), (
        f"Sample count mismatch in {split}"
    )

    max_length = 0

    for index, (word_sample, token_sample) in enumerate(
        zip(word_split, token_split)
    ):
        input_ids = token_sample["input_ids"]
        attention_mask = token_sample["attention_mask"]
        labels = token_sample["labels"]

        # Check 1: input_ids, attention_mask and labels have the same length
        assert len(input_ids) == len(attention_mask) == len(labels), (
            f"Length mismatch in {split} at sample {index}"
        )

        # Check 2: attention_mask contains only 1 (no padding before batching)
        assert all(mask == 1 for mask in attention_mask), (
            f"Unexpected attention_mask value in {split} at sample {index}"
        )

        # Check 3: sequence starts with [CLS] and ends with [SEP], both ignored
        assert input_ids[0] == tokenizer.cls_token_id, (
            f"Missing [CLS] in {split} at sample {index}"
        )
        assert input_ids[-1] == tokenizer.sep_token_id, (
            f"Missing [SEP] in {split} at sample {index}"
        )
        assert labels[0] == IGNORE_LABEL_ID and labels[-1] == IGNORE_LABEL_ID, (
            f"Special tokens not ignored in {split} at sample {index}"
        )

        # Check 4: non-ignored labels must exactly match the word-level labels
        kept_labels = [label for label in labels if label != IGNORE_LABEL_ID]

        assert kept_labels == word_sample["ner_tags_3class"], (
            f"Label alignment mismatch in {split} at sample {index}"
        )

        max_length = max(max_length, len(input_ids))

    print(
        f"{split:<12} | "
        f"Samples: {len(token_split)} | "
        f"Max tokens: {max_length}"
    )

print("\nTokenization validation completed successfully!")