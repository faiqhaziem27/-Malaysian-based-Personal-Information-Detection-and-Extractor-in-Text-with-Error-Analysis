"""
Print a Markdown results table from a training run's run_info.json.

Run from the project root:
    python -m model.generate_table <run_name>
    python -m model.generate_table models/checkpoints/<run_name>/run_info.json

Works with both formats:
    - new runs (English + Malay): one row per dataset and split
    - English-only runs: validation and test rows
"""

import json
import sys
from pathlib import Path

CHECKPOINT_DIR = Path("models/checkpoints")
METRICS = ["precision", "recall", "f1", "accuracy"]


def find_run_info(argument):
    # Accept a run name, a run folder, or a direct path to run_info.json.
    path = Path(argument)
    if path.suffix == ".json":
        return path
    if path.is_dir():
        return path / "run_info.json"
    return CHECKPOINT_DIR / argument / "run_info.json"


def format_pct(value):
    # Turn decimals (0.9465) into percentages (94.65%).
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "n/a"


def format_loss(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def table_row(label, metrics, prefix):
    # metrics keys look like "<prefix>_f1", "<prefix>_loss", ...
    cells = [format_pct(metrics.get(f"{prefix}_{name}")) for name in METRICS]
    cells.append(format_loss(metrics.get(f"{prefix}_loss")))
    return f"| **{label}** | " + " | ".join(cells) + " |"


def build_rows(data):
    # New format: combined validation plus one entry per dataset and split.
    if "per_dataset_metrics" in data:
        rows = [table_row("Combined validation",
                          data.get("combined_validation_metrics", {}), "eval")]
        labels = {
            "english_validation": "English validation",
            "english_test": "English test",
            "malay_validation": "Malay validation",
            "malay_test": "Malay test",
        }
        for set_name, label in labels.items():
            metrics = data["per_dataset_metrics"].get(set_name, {})
            rows.append(table_row(label, metrics, set_name))
        return rows

    # Older format: a single validation and test result
    return [
        table_row("Validation", data.get("validation_metrics", {}), "eval"),
        table_row("Test", data.get("test_metrics", {}), "test"),
    ]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m model.generate_table <run_name>")

    path = find_run_info(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"Could not find '{path}'.")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"## Evaluation Results: {data.get('run_name', path.parent.name)}\n")
    print("| Split | Precision | Recall | F1-Score | Accuracy | Loss |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in build_rows(data):
        print(row)


if __name__ == "__main__":
    main()