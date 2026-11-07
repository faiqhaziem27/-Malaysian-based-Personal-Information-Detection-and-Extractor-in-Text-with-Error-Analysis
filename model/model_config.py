from dataclasses import dataclass, field
from pathlib import Path

from src.label_config import LABEL2ID, ID2LABEL, NUM_LABELS


@dataclass
class ModelConfig:

    # Model settings
    pretrained_model_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 128

    # Label settings
    num_labels: int = NUM_LABELS
    label2id: dict = field(default_factory=lambda: LABEL2ID.copy())
    id2label: dict = field(default_factory=lambda: ID2LABEL.copy())

    # Training settings
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01

    # Output settings
    output_dir: Path = Path("models/checkpoints")
