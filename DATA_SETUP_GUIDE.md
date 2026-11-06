# Dataset Setup Guide

This guide explains how to download, prepare, and load the dataset for our NER project.

## Step 1: Clone the Repository

``` bash
git clone --branch data-engineering --single-branch git@github.com:faiqhaziem27/-Malaysian-based-Personal-Information-Detection-and-Extractor-in-Text-with-Error-Analysis.git airost-project
```

**Meaning:** Downloads the repository's `data-engineering` branch into a local folder named `airost-project`.

## Step 2: Enter the Project Folder

``` bash
cd airost-project
```

**Meaning:** Moves into the project directory.

## Step 3: Create and Activate a Python Environment

``` bash
conda create -n ner_data python=3.11
conda activate ner_data
```

**Meaning:**

- Creates a Conda environment named `ner_data` with Python 3.11.
- Activates the environment for this project.

## Step 4: Install Dependencies

``` bash
pip install -r requirements.txt
```

**Meaning:** Installs the required Python packages listed in `requirements.txt`.

## Step 5: Download the Raw Dataset

``` bash
python src/download_dataset.py
```

**Meaning:** Downloads the original CoNLL-2003 dataset from Hugging Face.

**Output location:**
`data/raw/conll2003/`

## Step 6: Prepare the Processed Dataset

``` bash
python src/prepare_dataset.py
```

**Meaning:** Processes the raw dataset and converts the original NER labels into our project's three target entity categories: Person, Organization, and Location.

**Output location:**
`data/processed/conll2003/`

## Step 7: Load and Verify the Dataset

``` bash
python src/load_processed_dataset.py
```

**Meaning:** Loads the processed dataset and displays its basic information for verification.

| Split | Examples |
| --- | ---: |
| Train | 14,041 |
| Validation | 3,250 |
| Test | 3,453 |

## Quick Reference

Run these commands **in order** after completing Steps 1–4:

``` bash
python src/download_dataset.py
python src/prepare_dataset.py
python src/load_processed_dataset.py
```

**Remember:**

- Raw and processed datasets are not uploaded to GitHub.
- Each teammate must download and prepare the data locally.
- Always follow the same preprocessing pipeline to ensure consistency across the team.