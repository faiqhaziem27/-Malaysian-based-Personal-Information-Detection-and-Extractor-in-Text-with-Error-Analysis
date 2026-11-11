# NER Tokenization & Label Alignment Guide

## 1. Overview

Script: `src/tokenize_dataset.py`

Pre-trained Transformer models do not process raw word strings directly. They split words into subword tokens and map them to vocabulary IDs. Because a single word can be split into multiple subwords, word-level NER labels (`ner_tags_3class`) must be aligned with the generated subword sequence before fine-tuning.

## 2. Dynamic Tokenization Design

Instead of saving static tokenized IDs to disk, `src/tokenize_dataset.py` performs tokenization dynamically in memory using Hugging Face `AutoTokenizer`:

- Default model checkpoint: `distilbert-base-cased`
- Any compatible Hugging Face model checkpoint can be passed via `model_checkpoint` without reprocessing the underlying dataset on disk.

## 3. Subword Label Alignment Rules

Alignment uses `tokenized_inputs.word_ids()` and PyTorch's default ignored loss index (`-100`):

| Token Type | Condition | Assigned Label | Reason |
|---|---|---|---|
| Special tokens (`[CLS]`, `[SEP]`) | `word_id is None` | `-100` | Boundary markers, not real text entities |
| First subword of a word | `word_id != previous_word_id` | Original label (`0`–`6`) | Preserves one loss contribution per word |
| Subsequent subword pieces | `word_id == previous_word_id` | `-100` | Prevents multi-piece words from being over-weighted |

### Example Alignment (`distilbert-base-cased`, Training Sample #2)

Original words: `["BRUSSELS", "1996-08-22"]` -> Word labels: `["B-LOC", "O"]` (`[5, 0]`)

| Subword | Input ID | Label ID | Label Name |
|---|---:|---:|---|
| `[CLS]` | 101 | -100 | IGN (-100) |
| `BR` | 26660 | 5 | B-LOC |
| `##US` | 13329 | -100 | IGN (-100) |
| `##SE` | 12649 | -100 | IGN (-100) |
| `##LS` | 15928 | -100 | IGN (-100) |
| `1996` | 1820 | 0 | O |
| `-` | 118 | -100 | IGN (-100) |
| `08` | 4775 | -100 | IGN (-100) |
| `-` | 118 | -100 | IGN (-100) |
| `22` | 1659 | -100 | IGN (-100) |
| `[SEP]` | 102 | -100 | IGN (-100) |

## 4. Output Dataset Schema

Raw columns (`id`, `tokens`, `ner_tags`, etc.) and `token_type_ids` are removed so the output works directly with both Hugging Face `Trainer` and custom PyTorch `DataLoader` loops using `DataCollatorForTokenClassification`:

| Feature | Description |
|---|---|
| `input_ids` | Subword vocabulary IDs |
| `attention_mask` | `1` for real tokens (`0` when padded in batches) |
| `labels` | Aligned NER label IDs (`0`–`6` and `-100`) |

## 5. Usage for Model Training

```python
from src.tokenize_dataset import load_tokenized_dataset

# Option 1: Use default checkpoint ('distilbert-base-cased')
tokenized_dataset, tokenizer = load_tokenized_dataset()

# Option 2: Specify another model checkpoint
# tokenized_dataset, tokenizer = load_tokenized_dataset("bert-base-cased")

train_dataset = tokenized_dataset["train"]
validation_dataset = tokenized_dataset["validation"]
test_dataset = tokenized_dataset["test"]
```

## 6. Verification

Run from the project root:

```bash
python src/tokenize_dataset.py
python src/validate_tokenization.py
```

- `tokenize_dataset.py`: Quick check of lengths and label IDs, and prints one sample alignment table.
- `validate_tokenization.py`: Full validation that compares tokenized labels against the original word-level labels.

### Validation Checks (`validate_tokenization.py`)

| Check | Description |
|---|---|
| Length | `input_ids`, `attention_mask` and `labels` have the same length |
| Attention mask | All values are `1` (no padding before batching) |
| Special tokens | Sequence starts with `[CLS]` and ends with `[SEP]`, both labeled `-100` |
| Label alignment | After removing `-100`, the remaining labels exactly match `ner_tags_3class` |

### Validation Results (`distilbert-base-cased`)

| Split | Samples | Max Tokens |
|---|---:|---:|
| Train | 14,041 | 173 |
| Validation | 3,250 | 151 |
| Test | 3,453 | 148 |

The longest training sequence is 173 tokens, well below the 512-token limit of DistilBERT. No CoNLL-2003 sample is truncated during training.

Long input documents (e.g., reports longer than 512 tokens) only need to be handled at prediction time.