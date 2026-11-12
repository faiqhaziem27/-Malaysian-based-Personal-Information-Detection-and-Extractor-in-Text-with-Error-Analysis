"""
Run the fine-tuned NER model on text.

Usage (from the project root):
    python -m src.infer "Ahmad bin Ali works at Petronas in Kuala Lumpur."
    python -m src.infer --file sample.txt
    python -m src.infer --model models/checkpoints/conll_v1 "Some text"

The model folder is read from --model, then the NER_MODEL_DIR environment
variable, then DEFAULT_MODEL_DIR.
"""

import argparse
import os
from functools import lru_cache

import torch
from transformers import pipeline

DEFAULT_MODEL_DIR = "models/checkpoints/run"


def get_model_dir(model_dir=None):
    """Resolve which model folder to load."""
    return model_dir or os.environ.get("NER_MODEL_DIR", DEFAULT_MODEL_DIR)


@lru_cache(maxsize=None)
def load_ner_pipeline(model_dir=None):
    """Load the model once and reuse it on later calls."""
    resolved = get_model_dir(model_dir)

    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"Model folder '{resolved}' not found. Unzip the model there or "
            "set NER_MODEL_DIR."
        )

    return pipeline(
        "token-classification",
        model=resolved,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )


def predict(text, model_dir=None):
    """
    Detect entities in text.

    Text is processed line by line, so long documents (for example PDF or OCR
    output) stay under the model's length limit. Character offsets refer to
    the original text.

    Returns a list of dicts: entity, type, score, start, end.
    """
    ner = load_ner_pipeline(model_dir)

    # Split into lines, remembering where each line starts in the full text
    lines = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            lines.append((offset, line.rstrip("\r\n")))
        offset += len(line)

    if not lines:
        return []

    results = ner([line for _, line in lines])

    # A single input can come back as a flat list, so normalise to one list per line
    if results and isinstance(results[0], dict):
        results = [results]

    entities = []
    for (line_start, _), line_entities in zip(lines, results):
        for ent in line_entities:
            entities.append({
                "entity": ent["word"],
                "type": ent["entity_group"],
                "score": round(float(ent["score"]), 3),
                "start": line_start + ent["start"],
                "end": line_start + ent["end"],
            })

    return entities


def print_table(entities):
    """Print entities as a simple text table."""
    if not entities:
        print("No entities found.")
        return

    width = max(len(e["entity"]) for e in entities)
    width = max(width, len("Entity"))

    print(f"{'Entity':<{width}} | {'Type':<4} | {'Score':<5} | Span")
    print("-" * (width + 28))
    for e in entities:
        span = f"{e['start']}-{e['end']}"
        print(f"{e['entity']:<{width}} | {e['type']:<4} | {e['score']:<5} | {span}")


def main():
    parser = argparse.ArgumentParser(description="Run NER on text.")
    parser.add_argument("text", nargs="?", help="Text to analyse")
    parser.add_argument("--file", help="Path to a UTF-8 text file")
    parser.add_argument("--model", help="Path to the model folder")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text or --file.")

    print_table(predict(text, args.model))


if __name__ == "__main__":
    main()