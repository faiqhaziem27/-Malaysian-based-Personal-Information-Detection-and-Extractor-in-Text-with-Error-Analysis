from transformers import AutoTokenizer
try:
    from load_processed_dataset import load_ner_dataset
    from label_config import ID2LABEL
    from project_paths import REQUIRED_SPLITS
except ModuleNotFoundError:
    from src.load_processed_dataset import load_ner_dataset
    from src.label_config import ID2LABEL
    from src.project_paths import REQUIRED_SPLITS

# Default pre-trained model checkpoint for tokenization
DEFAULT_MODEL_CHECKPOINT = "distilbert-base-cased"

# Label ID ignored by PyTorch CrossEntropyLoss during training
IGNORE_LABEL_ID = -100


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize word-level inputs and align NER labels with subword tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_token_type_ids=False
    )

    aligned_labels = []

    for index, word_labels in enumerate(examples["ner_tags_3class"]):
        word_ids = tokenized_inputs.word_ids(batch_index=index)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            # Special tokens like [CLS] and [SEP] have a word_id of None
            if word_id is None:
                label_ids.append(IGNORE_LABEL_ID)
            # Assign the original label to the first subword of each word
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            # Assign -100 to subsequent subword pieces of the same word
            else:
                label_ids.append(IGNORE_LABEL_ID)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def load_tokenized_dataset(model_checkpoint=DEFAULT_MODEL_CHECKPOINT):
    """
    Load the processed dataset and apply subword tokenization dynamically.

    Args:
        model_checkpoint (str): Hugging Face model checkpoint name.

    Returns:
        tuple: (tokenized_dataset, tokenizer)
    """
    dataset = load_ner_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    return tokenized_dataset, tokenizer


if __name__ == "__main__":
    print(f"Loading and tokenizing dataset with '{DEFAULT_MODEL_CHECKPOINT}'...\n")
    tokenized_dataset, tokenizer = load_tokenized_dataset()

    print(tokenized_dataset)

    # Verify token-label alignment across all splits
    print("\nAlignment Verification:\n")

    for split in REQUIRED_SPLITS:
        split_data = tokenized_dataset[split]

        for index, sample in enumerate(split_data):
            assert (
                len(sample["input_ids"])
                == len(sample["attention_mask"])
                == len(sample["labels"])
            ), f"Length mismatch in {split} at sample {index}"

            for label_id in sample["labels"]:
                assert label_id == IGNORE_LABEL_ID or label_id in ID2LABEL, (
                    f"Invalid label ID {label_id} in {split} at sample {index}"
                )

        print(f"{split:<12} | Verified {len(split_data)} samples")

    # Select a sample showing subword entity splitting (fallback to sample 0)
    sample_to_show = tokenized_dataset["train"][0]
    sample_index = 0

    for idx in range(min(10, len(tokenized_dataset["train"]))):
        candidate = tokenized_dataset["train"][idx]
        labels = candidate["labels"]
        has_split_entity = any(
            labels[i] > 0 and labels[i + 1] == IGNORE_LABEL_ID
            for i in range(len(labels) - 2)
        )
        if has_split_entity:
            sample_to_show = candidate
            sample_index = idx
            break

    subword_tokens = tokenizer.convert_ids_to_tokens(sample_to_show["input_ids"])

    print(f"\nTraining Sample #{sample_index} Subword Alignment:\n")
    print(f"{'Subword':<16} | {'Input ID':<8} | {'Label ID':<8} | {'Label Name'}")
    print("-" * 52)

    for subword, input_id, label_id in zip(
        subword_tokens,
        sample_to_show["input_ids"],
        sample_to_show["labels"]
    ):
        label_name = "IGN (-100)" if label_id == IGNORE_LABEL_ID else ID2LABEL[label_id]
        print(f"{subword:<16} | {input_id:<8} | {label_id:<8} | {label_name}")