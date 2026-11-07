# from .model_config import ModelConfig

# from src.tokenize_dataset import load_tokenized_dataset

# Defaults to "distilbert-base-cased", or pass any other model name we choose
# # tokenized_dataset, tokenizer = load_tokenized_dataset(config.pretrained_model_name)

# train_dataset = tokenized_dataset["train"]
# validation_dataset = tokenized_dataset["validation"]
# test_dataset = tokenized_dataset["test"]

##############

"""
Fine-tune a token classification model for NER.

Run from the project root:
    python -m train
"""

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from model.create_model import create_model
from model.model_config import ModelConfig
from src.tokenize_dataset import IGNORE_LABEL_ID, load_tokenized_dataset

SEED = 42


def build_compute_metrics(id2label):
    """Return a compute_metrics function that ignores -100 positions."""

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


def main():
    set_seed(SEED)
    config = ModelConfig()

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

    print("\nValidation results:")
    print(trainer.evaluate(tokenized_dataset["validation"]))

    print("\nTest results:")
    print(trainer.predict(tokenized_dataset["test"]).metrics)

    final_dir = config.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved best model and tokenizer to {final_dir}")


if __name__ == "__main__":
    main()