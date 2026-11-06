# NER Project - Data Documentation

## 1. Dataset Information

- Dataset Name: CoNLL-2003
- Source: Hugging Face
- Repository: https://huggingface.co/datasets/tomaarsen/conll2003
- Task: Named Entity Recognition (NER)
- Language: English

## 2. Dataset Splits

| Split | Number of Samples |
|---|---:|
| Training | 14,041 |
| Validation | 3,250 |
| Testing | 3,453 |
| Total | 20,744 |

## 3. Original Dataset Labels

The original dataset contains 9 BIO labels:

- O
- B-PER
- I-PER
- B-ORG
- I-ORG
- B-LOC
- I-LOC
- B-MISC
- I-MISC

## 4. Processed Dataset Labels

Our project uses 7 labels:

| ID | Label |
|---:|---|
| 0 | O |
| 1 | B-PER |
| 2 | I-PER |
| 3 | B-ORG |
| 4 | I-ORG |
| 5 | B-LOC |
| 6 | I-LOC |

## 5. Data Transformation

- Original B-MISC is converted to O.
- Original I-MISC is converted to O.
- PER, ORG and LOC labels remain unchanged.
- Original ner_tags are preserved.
- New labels are stored in ner_tags_3class.

## 6. Dataset Columns

| Column | Description |
|---|---|
| id | Sample ID |
| document_id | Document ID |
| sentence_id | Sentence ID |
| tokens | Word-level tokens |
| pos_tags | Part-of-speech labels |
| chunk_tags | Chunk labels |
| ner_tags | Original NER labels |
| ner_tags_3class | Processed NER labels |

## 7. Dataset Locations

- Raw Dataset: data/raw/conll2003/
- Processed Dataset: data/processed/conll2003_3class/

## 8. Dataset Validation

- All original dataset splits are preserved.
- Token-level label transformation was checked.
- Dataset sample counts were verified.
- Processed dataset contains 7 NER labels.

## 9. Model Training Notes

The processed dataset is intended for token classification fine-tuning.

The model developer must:

- Use the shared label_config.py.
- Use the tokens column as input.
- Use ner_tags_3class as the word-level target labels.
- Align labels with the selected model tokenizer.
