"""
Run the fine-tuned NER model on text.

Usage (from the project root):
    python -m src.infer "Ahmad bin Ali works at Petronas in Kuala Lumpur."
    python -m src.infer --file sample.txt
    python -m src.infer --model models/checkpoints/conll_v1 "Some text"
    python -m src.infer --redact "Ahmad bin Ali works at Petronas."
    python -m src.infer --redact --style mask "Ahmad bin Ali works at Petronas."
    python -m src.infer --redact --placeholder "[{type} {score:.2f}]" --min-score 0.9 "Some text"

The model folder is read from --model, then the NER_MODEL_DIR environment
variable, then DEFAULT_MODEL_DIR.
"""

import argparse
import os
from functools import lru_cache

import torch
from transformers import pipeline

DEFAULT_MODEL_DIR = "VeeJack/malaysian-ner-distilbert"
DEFAULT_PLACEHOLDER = "[{type}]"
MAX_CHUNK_TOKENS = 400 


def get_model_dir(model_dir=None):
    """Resolve which model folder to load."""
    return model_dir or os.environ.get("NER_MODEL_DIR", DEFAULT_MODEL_DIR)


@lru_cache(maxsize=None)
def load_ner_pipeline(model_dir=None):
    """Load the model once and reuse it on later calls."""
    resolved = get_model_dir(model_dir)

    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"Model folder '{resolved}' is a local folder, loading it from the Hugging Face Hub..."
        )

    return pipeline(
        "token-classification",
        model=resolved,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

def find_break(line, offsets, start_token, end_token):
    """
    Choose the token index where the next chunk should start.

    Searches backwards from end_token for the start of a word, preferring the
    start of a sentence, so entities are not cut in half. If no word start is
    found in the second half of the chunk, falls back to a hard split.
    """
    lowest = start_token + (end_token - start_token) // 2
    word_start = None

    for index in range(end_token, lowest - 1, -1):
        if index <= start_token:
            break

        first_char = offsets[index][0]
        if first_char == 0 or not line[first_char - 1].isspace():
            continue  # this token continues a word, so it is not a safe break

        previous_end = offsets[index - 1][1]
        if line[previous_end - 1] in ".!?":
            return index  # sentence boundary, the best place to cut

        if word_start is None:
            word_start = index

    return word_start if word_start is not None else end_token

def split_long_line(line, tokenizer, max_tokens=MAX_CHUNK_TOKENS):
    """
    Split a line into chunks that fit the model.

    Returns a list of (character_offset, chunk_text). The chunks cover the line
    exactly, so joining them gives back the original text. A short line comes
    back as a single chunk.
    """
    encoding = tokenizer(line, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    total = len(offsets)

    if total <= max_tokens:
        return [(0, line)]

    chunks = []
    chunk_start = 0
    token = 0

    while token < total:
        end_token = min(token + max_tokens, total)
        if end_token < total:
            end_token = find_break(line, offsets, token, end_token)

        end_char = len(line) if end_token >= total else offsets[end_token][0]
        chunks.append((chunk_start, line[chunk_start:end_char]))

        chunk_start = end_char
        token = end_token

    return chunks

def predict(text, model_dir=None):
    """
    Detect entities in text.

    Text is processed line by line, and any line longer than the model can read
    is split into chunks first, so nothing is silently cut off. Character
    offsets always refer to the original text.

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

    # Cut over-long lines into chunks, keeping each chunk's position in the text
    pieces = []
    for line_start, line in lines:
        for chunk_start, chunk in split_long_line(line, ner.tokenizer):
            pieces.append((line_start + chunk_start, chunk))

    results = ner([chunk for _, chunk in pieces])

    # A single input can come back as a flat list, so normalise to one list per chunk
    if results and isinstance(results[0], dict):
        results = [results]

    entities = []
    for (piece_start, _), piece_entities in zip(pieces, results):
        for ent in piece_entities:
            entities.append({
                "entity": ent["word"],
                "type": ent["entity_group"],
                "score": round(float(ent["score"]), 3),
                "start": piece_start + ent["start"],
                "end": piece_start + ent["end"],
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


def redact_text(text, entities, placeholder="[{type}]", min_score=0.0, style="mask"):
    """
    Redact the text by replacing each entity with a placeholder.

    The placeholder can include {type} and {score} fields. For example:
        "REDACTED" -> "REDACTED"
        "{type}" -> "PER"
        "{type}:{score:.2f}" -> "PER:0.95"

    Entities are replaced in reverse order so that earlier replacements don't
    change the character offsets of later ones.
    """
    redacted = text
    for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
        if entity["score"] < min_score:
            continue
        if style == "label":
            replacement = placeholder.format(type=entity["type"], score=entity["score"])
        elif style == "mask":
            replacement = "█" * (entity["end"] - entity["start"])
        else:
            raise ValueError(f"Unknown style: {style}")
        redacted = (
            redacted[: entity["start"]] + replacement + redacted[entity["end"] :]
        )
    return redacted


def main():
    parser = argparse.ArgumentParser(description="Run NER on text.")
    parser.add_argument("text", nargs="?", help="Text to analyse")
    parser.add_argument("--file", help="Path to a UTF-8 text file")
    parser.add_argument("--model", help="Path to the model folder")

    # argument for redaction
    parser.add_argument("--redact", action="store_true", help="Print the text with entities replaced instead of a table")
    parser.add_argument("--style", choices=["label", "mask"], default="label", help="label: [PER]   mask: ********")
    parser.add_argument("--placeholder", default="[{type}]", help="Label format using {type} and {score}")
    parser.add_argument("--min-score", type=float, default=0.0, help="Only redact entities at or above this confidence")    
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text or --file.")

    entities = print_table(predict(text, args.model))
    if args.redact:
        print(redact_text(text, entities, args.placeholder, args.min_score, args.style))
    else:
        print_table(entities)

if __name__ == "__main__":
    main()