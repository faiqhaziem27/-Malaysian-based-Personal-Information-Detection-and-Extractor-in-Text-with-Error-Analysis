# Malaysian NER Dataset Guide

## 1. Overview

This custom dataset addresses **Bonus 2** of the project problem statement by providing labeled Malaysian text across three target domains:

- **Universities** (e.g., `Universiti Malaya`, `Universiti Teknologi Malaysia`, `Universiti Sains Malaysia`)
- **Hospitals & Healthcare** (e.g., `Hospital Kuala Lumpur`, `Sunway Medical Centre`, `Institut Jantung Negara`)
- **Companies & Public Institutions** (e.g., `Maybank`, `Petronas`, `Tenaga Nasional Berhad`, `Bank Negara Malaysia`)

All samples are synthetic incident reports and customer support chat sentences containing local Malaysian personal names (`PER`), organizations (`ORG`), and locations (`LOC`), along with negative examples containing no entities (`O`).

## 2. Dataset Locations

| Type | Path | Git Status |
|---|---|---|
| Raw annotated JSON | `data/malaysia/malaysia_ner_samples.json` | Tracked in Git |
| Processed `DatasetDict` | `data/processed/malaysia_3class/` | Ignored (`.gitignore`) |

## 3. Dataset Splits & Distribution

| Split | Samples | `PER` Entities | `ORG` Entities | `LOC` Entities | Max Subword Tokens (`distilbert-base-cased`) |
|---|---:|---:|---:|---:|---:|
| Train | 70 | 67 | 62 | 65 | 38 |
| Validation | 15 | 15 | 13 | 13 | 36 |
| Test | 15 | 15 | 13 | 14 | 31 |
| **Total** | **100** | **97** | **88** | **92** | **38** |

### Training Split BIO Label Distribution

| Label | ID | Count |
|---|---:|---:|
| `O` | 0 | 714 |
| `B-PER` | 1 | 67 |
| `I-PER` | 2 | 121 |
| `B-ORG` | 3 | 62 |
| `I-ORG` | 4 | 105 |
| `B-LOC` | 5 | 65 |
| `I-LOC` | 6 | 36 |

## 4. Setup & Verification

Run from the project root to build and validate the Malaysian dataset locally:

```bash
python src/prepare_malaysia_dataset.py
python src/load_malaysia_dataset.py
```

- `prepare_malaysia_dataset.py`: Converts the raw JSON annotations into a Hugging Face `DatasetDict` at `data/processed/malaysia_3class/`.
- `load_malaysia_dataset.py`: Validates word-level lengths, BIO tag transitions, and subword label alignment across all splits.

## 5. Usage for Model Training & Evaluation

```python
from src.load_malaysia_dataset import (
    load_malaysia_dataset,
    load_tokenized_malaysia_dataset
)

# 1. Load word-level dataset ('id', 'tokens', 'ner_tags_3class')
word_dataset = load_malaysia_dataset()

# 2. Load tokenized dataset ('input_ids', 'attention_mask', 'labels')
tokenized_dataset, tokenizer = load_tokenized_malaysia_dataset()

train_dataset = tokenized_dataset["train"]
validation_dataset = tokenized_dataset["validation"]
test_dataset = tokenized_dataset["test"]
```

Note: `load_tokenized_malaysia_dataset()` shares `DEFAULT_MODEL_CHECKPOINT` from `src/tokenize_dataset.py`. Updating `DEFAULT_MODEL_CHECKPOINT` in `src/tokenize_dataset.py` automatically switches the tokenizer for both CoNLL-2003 and the Malaysian dataset.