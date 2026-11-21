# NER Project - Model Documentation

## 1. Model Overview and Information
- **Developed by:** TheSilenced
- Pretrained Base Model: [`distilbert/distilbert-base-multilingual-cased`](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- Architecture: Distilled Transformer Encoder (6 layers, 768 hidden size, 12 attention heads, ~134M parameters)
- Source: Hugging Face / Pytorch
- Repository: https://huggingface.co/distilbert/distilbert-base-multilingual-cased
- Task: Named Entity Recognition (NER)
- Languages: English('en'), Malay('ms')

## 2. Model Configuration
* **Pretrained Base Model:** `distilbert-base-multilingual-cased`
* **Maximum Sequence Length:** `128` tokens
* **Number of Labels (`num_labels`):** `7`
* **Labels:** `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`

## 3. Training Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Batch Size** | `16` |
| **Learning Rate** | `2e-05` |
| **Epochs** | `3` |
| **Weight Decay** | `0.01` |
| **Output Directory** | `models/checkpoints` |

## 4. Training Data and Preprocessing
- **Dataset Card:** For a complete breakdown on data documentation and setup, refer to the following:
    - [Data Documentation](./DATA_DOCUMENTATION.md)
    - [Data Setup Guide](./DATA_SETUP_GUIDE.md)
    - [Malaysia Dataset Guide](./MALAYSIA_DATASET_GUIDE.md)
    - [Dataset Usage](./DATASET_USAGE.md)
    - [Tokenization Guide](./TOKENIZATION_GUIDE.md)
- **Corpus Summary:** Fine-tuned on a combined bilingual corpus of English and Malay text comprised of domain-specific named entities.

## 5. Evaluation Results

| Split | Precision | Recall | F1-Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Validation** | 94.66% | 95.28% | 94.97% | 99.21% |
| **Test** | 91.28% | 91.65% | 91.46% | 98.43% |

- Note that the evaluation results are based on training using the Conll_v1 dataset. Final results are subject to change.

## 6. Setup

### Requirements
- Python 3.10 or newer
- Install the dependencies from the project root:
```bash
pip install -r requirements.txt
pip install torch seqeval accelerate numpy
```
- The model needs `transformers`, `datasets`, `torch`, `numpy` and `seqeval`. 
- A GPU is only needed for training (for example, a free Google Colab T4). Running the trained model on a few sentences works on a CPU.

Open [ner_training.ipynb](./notebooks/ner_training.ipynb) using your preferred platform (i.e Google Colab, Kaggle) and run all the cells in order.

### Getting the model weights
The trained weights are not stored in the repository (`models/` is listed in `.gitignore`), so download them separately:

- Download link: `https://drive.google.com/file/d/12c7HzSjl8Axs90nWUDcXSsrbEOmreGM0/view?usp=sharing`


Extract the files into `models/checkpoints/<run_name>/` so that the folder contains the model and tokenizer files directly:

```
models/checkpoints/<run_name>/
├── config.json
├── model.safetensors
├── tokenizer files (tokenizer.json, tokenizer_config.json, ...)
└── run_info.json
```

The model file is several hundred MB. If the zip was created from `models/checkpoints/<run_name>`, extracting it at the project root recreates this path. The folder you point the code at must be the one that has `config.json` directly inside it.

### Checking that the model loads
Run this from the project root, replacing `<run_name>` with your folder name:

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="models/checkpoints/<run_name>",
    aggregation_strategy="simple",
)

for entity in ner("Ahmad bin Ali works at Petronas in Kuala Lumpur."):
    print(entity["entity_group"], entity["word"], round(float(entity["score"]), 3))
```

It should print the detected people (`PER`), organizations (`ORG`) and locations (`LOC`) with a confidence score. The model only detects these three types.

### Training a new model
- The datasets are not stored in the repository. Prepare them first by following the [Data Setup Guide](./DATA_SETUP_GUIDE.md).
- Train from the project root, using a run name that has not been used before:
```bash
python -m model.train <run_name>
```

There are multiple training scripts with their dedicated purpose:
- [train.py](./model/train.py): combines both the English and Malay datasets
- [malay_train.py](./model/malay_train.py): trains the model with the Malay dataset only
- [english_train.py](./model/english_train.py): trains the model with the English dataset only

- The best checkpoint, the tokenizer and `run_info.json` are saved to `models/checkpoints/<run_name>/`. The script stops immediately if that folder already exists, so earlier runs are never overwritten.

### Model Training Notes:

- For every training run in [train.py](./model/train.py), a JSON file named run_info.json will be created consisting of the evaluation results.
- You can compare the results with your own script, or use [generate_table.py](./model/generate_table.py) to generate a text-based table.

**How to use generate_table.py:**

- As models\checkpoints\run is ignored in .gitignore, manually move **generate_table.py** to the installed model folder with the following command:
```bash
python -m model.generate_table <run_name>
```

- To run the script:
```bash
python models/checkpoints/run/generate_table.py
```
