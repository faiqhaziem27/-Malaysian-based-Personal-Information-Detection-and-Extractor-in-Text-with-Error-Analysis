"""
Fine-tune a token classification model for NER.

Run based on the given command:
    python -m model.train <run_name>
"""

# import necessary libraries
import json
import sys
from datetime import datetime

import numpy as np
import torch
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
from src.tokenize_dataset import IGNORE_LABEL_ID, load_tokenized_dataset

SEED = 42

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


def main():
    set_seed(SEED)
    config = ModelConfig()

    # Check if the output directory for this run already exists
    final_dir = config.output_dir / RUN_NAME
    if final_dir.exists():
        raise SystemExit(f"'{final_dir}' already exists. Pick a new run name.") # double check to avoid overwriting previous runs

    # Tokenizer and model must come from the same checkpoint
    tokenized_dataset, tokenizer = load_tokenized_dataset(config.pretrained_model_name)
    model = create_model(config)

    if tokenizer.name_or_path != config.pretrained_model_name:
        raise ValueError(
            f"Tokenizer '{tokenizer.name_or_path}' != model '{config.pretrained_model_name}'"
        )
    if len(tokenizer) > model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({len(tokenizer)}) exceeds model vocab ({model.config.vocab_size})"
        )
    if model.config.num_labels != len(config.label2id):
        raise ValueError("Model num_labels does not match label2id")

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
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,  # use tokenizer=tokenizer on older transformers
        compute_metrics=build_compute_metrics(config.id2label),
    )

    trainer.train()

    # Evaluate on validation and test sets
    validation_metrics = trainer.evaluate(tokenized_dataset["validation"])
    print("\nValidation results:")
    print(validation_metrics)

    test_metrics = trainer.predict(tokenized_dataset["test"]).metrics
    print("\nTest results:")
    print(test_metrics)

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
            split: len(tokenized_dataset[split]) for split in tokenized_dataset
        },
        "hyperparameters": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "weight_decay": config.weight_decay,
            "seed": SEED,
        },
        "validation_metrics": to_plain_dict(validation_metrics),
        "test_metrics": to_plain_dict(test_metrics),
    }

    metrics_path = final_dir / "run_info.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nSaved best model and tokenizer to {final_dir}")
    print(f"Saved run info to {metrics_path}")

if __name__ == "__main__":
    main()