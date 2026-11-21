
# NER Dataset Usage Guide

## 1. Dataset Overview

Dataset: CoNLL-2003
Task: Named Entity Recognition (NER)
Language: English

This dataset has been processed for our project
to recognize three entity types:

- PER: Person
- ORG: Organization
- LOC: Location

## 2. Dataset Location

Processed dataset:

data/processed/conll2003_3class/

## 3. Dataset Splits

| Split | Number of Samples |
|---|---:|
| Train | 14,041 |
| Validation | 3,250 |
| Test | 3,453 |

## 4. Dataset Loading

Use the shared loading function:

```python
from src.load_processed_dataset import load_ner_dataset

dataset = load_ner_dataset()

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]
```

## 5. Important Columns

| Column | Description |
|---|---|
| tokens | Original word-level tokens |
| ner_tags_3class | Processed word-level NER labels |
| ner_tags | Original 9-class NER labels |

For model training, use:

- Input: tokens
- Target: ner_tags_3class

The original ner_tags column should not be used
as the target for our 7-label project configuration.

## 6. Label Configuration

Import the shared label configuration:

```python
from src.label_config import LABEL_LIST, LABEL2ID, ID2LABEL
```

Do not redefine the label mapping independently.

## 7. Important Note: Tokenization

The dataset on disk contains word-level tokens
and word-level NER labels.

Before fine-tuning, use `src/tokenize_dataset.py`
to dynamically tokenize inputs and align subword labels.
See `TOKENIZATION_GUIDE.md` for details.

Do not assume that one word corresponds to
exactly one model token.

## 8. Dataset Loading Test

Run from the project root:

```bash
python src/load_processed_dataset.py
```

Expected split sizes:

- Train: 14,041
- Validation: 3,250
- Test: 3,453
