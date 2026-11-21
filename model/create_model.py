from transformers import AutoModelForTokenClassification

from .model_config import ModelConfig


def create_model(config: ModelConfig):
    # Load the pre-trained model for token classification with the specified configuration
    model = AutoModelForTokenClassification.from_pretrained(
        config.pretrained_model_name,
        num_labels=config.num_labels,
        label2id=config.label2id,
        id2label=config.id2label
    )

    return model

def main():
    config = ModelConfig()
    model = create_model(config)
    print(model)
    print(f"\nLabels: {model.config.id2label}")

if __name__ == "__main__":
    main()