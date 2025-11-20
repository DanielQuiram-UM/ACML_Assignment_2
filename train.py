from training_visualization import TrainingVisualizer
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json

from data_loader import DataLoader
from dataset import CIFAR10Dataset
from autoencoder import ConvAutoencoder
import random
import numpy as np
import tensorflow as tf
import os


# Set random seeds for reproducibility
# ---------------------------
SEED = 45
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Optional: enforce deterministic GPU operations (may slow down training)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("\n=== Hydra Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------
    # Use Hydra runtime output dir or fallback to "./results"
    output_dir = Path(hydra.utils.get_original_cwd()) / "results" / cfg.model.name.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # Save full config to YAML
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # --------------------------------------------------
    # Load CIFAR-10
    # --------------------------------------------------
    data_dir = cfg.data.data_dir
    loader = DataLoader(data_dir)
    x, y = loader.load_all_data()

    dataset = CIFAR10Dataset(x, y)
    dataset.normalize()
    x_train, x_val, x_test = dataset.split()

    # --------------------------------------------------
    # Build autoencoder from Hydra architecture config
    # --------------------------------------------------
    cae = ConvAutoencoder(
        input_shape=tuple(cfg.model.input_shape),
        architecture=cfg.model.architecture
    )

    # --------------------------------------------------
    # Train the model
    # --------------------------------------------------
    history = cae.train(
        x_train,
        x_val,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
    )

    # --------------------------------------------------
    # Save training loss history
    # --------------------------------------------------
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    print(f"Training history saved to: {history_path}")

    # --------------------------------------------------
    # Evaluate model on test set
    # --------------------------------------------------
    test_loss = cae.evaluate(x_test)
    with open(output_dir / "test_loss.txt", "w") as f:
        f.write(f"{test_loss}\n")
    print(f"Test loss saved to: {output_dir / 'test_loss.txt'}")

    # Optionally save model weights
    model_path = output_dir / "model_weights.weights.h5"
    cae.model.save_weights(model_path)
    print(f"Model weights saved to: {model_path}")

    return history


if __name__ == "__main__":
    main()

    results_root = Path("./results")

    # List of models to nclude in the plot
    include_models = ["baseline", "larger_kernel_size", "smaller_kernel_size"]

    # Only include the directories you listed
    results_dirs = [results_root / model_name for model_name in include_models if
                    (results_root / model_name / "history.json").exists()]

    if results_dirs:
        TrainingVisualizer.plot_multiple_from_dirs(
            results_dirs,
            save_path=results_root / "selected_models_loss.png"
        )

        # Optional: rank the selected models
        #TrainingVisualizer.rank_losses_from_dirs(
        #    results_dirs,
        #    use_val_loss=True
        #)
    else:
        print("No training histories found in results folder.")
