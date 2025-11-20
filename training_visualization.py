import matplotlib.pyplot as plt
import json
from pathlib import Path


class TrainingVisualizer:
    def __init__(self, history=None, history_path=None):
        """
        Initialize with either:
        - a Keras History object (history)
        - a path to a saved JSON history file (history_path)
        """
        if history is not None:
            self.history = history.history
        elif history_path is not None:
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            raise ValueError("Either history or history_path must be provided.")

def plot_loss(self, title="Training and Validation Loss over Epochs", save_path=None):
    """
    Plot training and validation loss for a single network.
    """
    epochs = range(1, len(self.history['loss']) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, self.history['loss'], label='Training Loss')
    if 'val_loss' in self.history:
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)

def plot_multiple_histories(histories_dict, title="Training and Validation Loss Comparison"):
    """
    Plot multiple training histories on the same figure for comparison.
    histories_dict: dictionary { 'Network Name': keras_history_object }
    """
    plt.figure(figsize=(8, 5))
    for name, history in histories_dict.items():
        plt.plot(history.history['loss'], label=f'{name} - Train')
        plt.plot(history.history['val_loss'], linestyle='--', label=f'{name} - Val')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    @staticmethod
    def plot_multiple_from_dirs(results_dirs, title="Training and Validation Loss Comparison", save_path=None):
        """
        Plot multiple training histories from saved JSON results directories.
        """
        plt.figure(figsize=(8,5))
        for dir_path in results_dirs:
            dir_path = Path(dir_path)
            model_name = dir_path.name
            history_path = dir_path / "history.json"
            if not history_path.exists():
                print(f"Warning: {history_path} does not exist, skipping.")
                continue
            with open(history_path, 'r') as f:
                history = json.load(f)
            epochs = range(1, len(history['loss']) + 1)
            plt.plot(epochs, history['loss'], label=f'{model_name} - Train')
            if 'val_loss' in history:
                plt.plot(epochs, history['val_loss'], linestyle='--', label=f'{model_name} - Val')

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which='both')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to: {save_path}")
        plt.show()

    @staticmethod
    def rank_losses_from_dirs(results_dirs, use_val_loss=True):
        """
        Print a ranking of models based on their final loss.
        results_dirs: list of directories containing history.json
        use_val_loss: if True, rank by final validation loss; else by training loss
        """
        losses = []

        for dir_path in results_dirs:
            dir_path = Path(dir_path)
            model_name = dir_path.name
            history_path = dir_path / "history.json"
            if not history_path.exists():
                print(f"Warning: {history_path} does not exist, skipping.")
                continue

            with open(history_path, 'r') as f:
                history = json.load(f)

            if use_val_loss and 'val_loss' in history:
                final_loss = history['val_loss'][-1]
            else:
                final_loss = history['loss'][-1]

            losses.append((model_name, final_loss))

        # Sort by loss ascending (best first)
        losses.sort(key=lambda x: x[1])

        print("\n=== Model Loss Ranking ===")
        for rank, (model_name, loss) in enumerate(losses, start=1):
            print(f"{rank}. {model_name}: {loss:.6f}")

        return losses

