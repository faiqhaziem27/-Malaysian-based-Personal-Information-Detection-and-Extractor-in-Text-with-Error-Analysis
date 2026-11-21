"""
Fine-tune a token classification model for NER on the English (CoNLL-2003)
and Malay datasets together.

Run based on the given command:
    python -m model.train <run_name>
"""

# import necessary libraries
import json
import sys
from datetime import datetime

import numpy as np
import torch
from datasets import concatenate_datasets
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# import from local modules
from model.create_model import create_model
from model.model_config import ModelConfig
from src.load_malaysia_dataset import load_tokenized_malaysia_dataset
from src.tokenize_dataset import IGNORE_LABEL_ID, load_tokenized_dataset

SEED = 42

# Repeat the Malay training data this many times (1 = no oversampling).
# Raise it if the Malay set is much smaller than CoNLL and gets drowned out.
OVERSAMPLE_MALAY = 1

MODEL_COLUMNS = ["input_ids", "attention_mask", "labels"]

# Get the run name from command line arguments, default to "run" if not provided
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "run"


def build_compute_metrics(id2label):
    # Return a compute_metrics function that ignores -100 positions.

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        true_labels = []
        true_predictions = []

        for pred_row, label_row in zip(predictions, labels):
            row_labels = []
            row_preds = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == IGNORE_LABEL_ID:
                    continue
                row_labels.append(id2label[int(label_id)])
                row_preds.append(id2label[int(pred_id)])
            true_labels.append(row_labels)
            true_predictions.append(row_preds)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics


def to_plain_dict(metrics):
    # Convert metric values (possibly numpy types) to plain Python floats.
    return {key: float(value) for key, value in metrics.items()}


def check_label_range(dataset, name, num_labels):
    # Every label must be -100 or a valid label ID, or training will break.
    for labels in dataset["labels"]:
        for label_id in labels:
            if label_id != IGNORE_LABEL_ID and not 0 <= label_id < num_labels:
                raise ValueError(
                    f"Invalid label ID {label_id} in {name} (expected 0-{num_labels - 1})"
                )


def main():
    set_seed(SEED)
    config = ModelConfig()

    # Fail fast: stop now if this run name is already taken
    final_dir = config.output_dir / RUN_NAME
    if final_dir.exists():
        raise SystemExit(f"'{final_dir}' already exists. Pick a new run name.")

    # Both datasets must be tokenized with the model's own checkpoint
    english, english_tokenizer = load_tokenized_dataset(config.pretrained_model_name)
    malay, malay_tokenizer = load_tokenized_malaysia_dataset(config.pretrained_model_name)
    tokenizer = english_tokenizer
    model = create_model(config)

    for name, tok in [("English", english_tokenizer), ("Malay", malay_tokenizer)]:
        if tok.name_or_path != config.pretrained_model_name:
            raise ValueError(
                f"{name} tokenizer '{tok.name_or_path}' != model "
                f"'{config.pretrained_model_name}'. Check that its loader "
                "accepts the checkpoint name."
            )
    if english_tokenizer.get_vocab() != malay_tokenizer.get_vocab():
        raise ValueError("English and Malay tokenizers have different vocabularies")
    if len(tokenizer) > model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({len(tokenizer)}) exceeds model vocab ({model.config.vocab_size})"
        )
    if model.config.num_labels != len(config.label2id):
        raise ValueError("Model num_labels does not match label2id")

    # Same columns and types, so the two datasets can be concatenated
    english = {split: english[split].select_columns(MODEL_COLUMNS) for split in english}
    malay = {split: malay[split].select_columns(MODEL_COLUMNS) for split in malay}
    for split in malay:
        malay[split] = malay[split].cast(english[split].features)

    for dataset_name, splits in [("english", english), ("malay", malay)]:
        for split in ("train", "validation"):
            check_label_range(splits[split], f"{dataset_name} {split}", config.num_labels)

    # Train on both together; validate on both together for checkpoint selection
    train_dataset = concatenate_datasets(
        [english["train"]] + [malay["train"]] * OVERSAMPLE_MALAY
    ).shuffle(seed=SEED)
    validation_dataset = concatenate_datasets(
        [english["validation"], malay["validation"]]
    )
    print(
        f"Training on {len(train_dataset)} sentences "
        f"({len(english['train'])} English + {len(malay['train'])} Malay x {OVERSAMPLE_MALAY})"
    )

    # Pads input_ids / attention_mask and pads labels with -100
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # use tokenizer=tokenizer on older transformers
        compute_metrics=build_compute_metrics(config.id2label),
    )

    trainer.train()

    # Combined validation score (the one used to pick the best checkpoint)
    validation_metrics = trainer.evaluate(validation_dataset)
    print("\nCombined validation results:")
    print(validation_metrics)

    # Score each dataset separately so one cannot hide the other
    evaluation_sets = {
        "english_validation": english["validation"],
        "english_test": english["test"],
        "malay_validation": malay["validation"],
        "malay_test": malay["test"],
    }
    per_dataset_metrics = {}
    for set_name, dataset in evaluation_sets.items():
        metrics = trainer.predict(dataset, metric_key_prefix=set_name).metrics
        per_dataset_metrics[set_name] = to_plain_dict(metrics)
        print(f"\n{set_name}:")
        print(metrics)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Record what this run was, next to the model
    run_info = {
        "run_name": RUN_NAME,
        "date": datetime.now().isoformat(timespec="seconds"),
        "base_model": config.pretrained_model_name,
        "labels": config.id2label,
        "num_labels": config.num_labels,
        "dataset_sizes": {
            "english": {split: len(english[split]) for split in english},
            "malay": {split: len(malay[split]) for split in malay},
            "combined_train": len(train_dataset),
            "oversample_malay": OVERSAMPLE_MALAY,
        },
        "hyperparameters": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "weight_decay": config.weight_decay,
            "seed": SEED,
        },
        "combined_validation_metrics": to_plain_dict(validation_metrics),
        "per_dataset_metrics": per_dataset_metrics,
    }

    metrics_path = final_dir / "run_info.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nSaved best model and tokenizer to {final_dir}")
    print(f"Saved run info to {metrics_path}")


if __name__ == "__main__":
    main()