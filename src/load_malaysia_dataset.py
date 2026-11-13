from collections import Counter
from datasets import load_from_disk
from transformers import AutoTokenizer
try:
    from project_paths import MALAYSIA_PROCESSED_DATASET_PATH, REQUIRED_SPLITS
    from label_config import ID2LABEL, LABEL_LIST
    from tokenize_dataset import (
        DEFAULT_MODEL_CHECKPOINT,
        IGNORE_LABEL_ID,
        tokenize_and_align_labels
    )
except ModuleNotFoundError:
    from src.project_paths import MALAYSIA_PROCESSED_DATASET_PATH, REQUIRED_SPLITS
    from src.label_config import ID2LABEL, LABEL_LIST
    from src.tokenize_dataset import (
        DEFAULT_MODEL_CHECKPOINT,
        IGNORE_LABEL_ID,
        tokenize_and_align_labels
    )


def load_malaysia_dataset():
    """
    Load the processed word-level Malaysian NER dataset.

    Returns:
        DatasetDict: Contains train, validation, and test splits.
    """
    if not MALAYSIA_PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed Malaysian dataset not found at: {MALAYSIA_PROCESSED_DATASET_PATH}. "
            "Please run 'python src/prepare_malaysia_dataset.py' first."
        )

    return load_from_disk(str(MALAYSIA_PROCESSED_DATASET_PATH))


def load_tokenized_malaysia_dataset(model_checkpoint=DEFAULT_MODEL_CHECKPOINT):
    """
    Load the processed Malaysian dataset and apply subword tokenization dynamically.

    Args:
        model_checkpoint (str): Hugging Face model checkpoint name.

    Returns:
        tuple: (tokenized_dataset, tokenizer)
    """
    dataset = load_malaysia_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    return tokenized_dataset, tokenizer


if __name__ == "__main__":
    word_dataset = load_malaysia_dataset()
    tokenized_dataset, tokenizer = load_tokenized_malaysia_dataset()

    # Count labels and complete entities in the training split
    label_counter = Counter()
    entity_counter = Counter()

    for sample in word_dataset["train"]:
        for tag_id in sample["ner_tags_3class"]:
            label_name = LABEL_LIST[tag_id]
            label_counter[label_name] += 1
            if label_name.startswith("B-"):
                entity_counter[label_name[2:]] += 1

    print("\nMalaysian Training Label Distribution:\n")
    for label_name, count in label_counter.most_common():
        print(f"{label_name:<10} : {count}")

    print("\nMalaysian Training Entity Count:\n")
    for entity_type, count in sorted(entity_counter.items()):
        print(f"{entity_type:<10} : {count}")

    # Validate word-level BIO transitions and subword alignment across all splits
    print("\nMalaysian Dataset & Tokenization Validation:\n")

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
            tokens = word_sample["tokens"]
            word_labels = word_sample["ner_tags_3class"]

            assert len(tokens) == len(word_labels), (
                f"Word token-label length mismatch in {split} at sample {index}"
            )

            # Check valid BIO transitions
            previous_entity = "O"
            for tag_id in word_labels:
                label_name = LABEL_LIST[tag_id]
                if label_name.startswith("I-"):
                    entity_type = label_name[2:]
                    assert previous_entity == entity_type, (
                        f"Invalid BIO transition to {label_name} in {split} at sample {index}"
                    )
                    previous_entity = entity_type
                elif label_name.startswith("B-"):
                    previous_entity = label_name[2:]
                else:
                    previous_entity = "O"

            # Check subword tokenization alignment
            input_ids = token_sample["input_ids"]
            attention_mask = token_sample["attention_mask"]
            labels = token_sample["labels"]

            assert len(input_ids) == len(attention_mask) == len(labels), (
                f"Subword length mismatch in {split} at sample {index}"
            )
            assert all(mask == 1 for mask in attention_mask), (
                f"Unexpected attention_mask value in {split} at sample {index}"
            )
            assert labels[0] == IGNORE_LABEL_ID and labels[-1] == IGNORE_LABEL_ID, (
                f"Special tokens not ignored in {split} at sample {index}"
            )

            kept_labels = [label for label in labels if label != IGNORE_LABEL_ID]
            assert kept_labels == word_labels, (
                f"Subword label alignment mismatch in {split} at sample {index}"
            )

            max_length = max(max_length, len(input_ids))

        print(
            f"{split:<12} | "
            f"Samples: {len(token_split)} | "
            f"Max tokens: {max_length}"
        )

    # Display first Malaysian training sample subword alignment
    sample_0 = tokenized_dataset["train"][0]
    subword_tokens = tokenizer.convert_ids_to_tokens(sample_0["input_ids"])

    print("\nFirst Malaysian Sample Subword Alignment:\n")
    print(f"{'Subword':<16} | {'Input ID':<8} | {'Label ID':<8} | {'Label Name'}")
    print("-" * 52)

    for subword, input_id, label_id in zip(
        subword_tokens,
        sample_0["input_ids"],
        sample_0["labels"]
    ):
        label_name = "IGN (-100)" if label_id == IGNORE_LABEL_ID else ID2LABEL[label_id]
        print(f"{subword:<16} | {input_id:<8} | {label_id:<8} | {label_name}")

    print("\nMalaysian dataset validation completed successfully!")