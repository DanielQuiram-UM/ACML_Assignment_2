# training_visualizer.py
import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self, history):
        """
        Initialize with a Keras History object.
        """
        self.history = history

    def plot_loss(self, title="Training and Validation Loss over Epochs"):
        """
        Plot training and validation loss for a single network.
        """
        plt.figure(figsize=(6,4))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)


def plot_multiple_histories(histories_dict, title="Training and Validation Loss Comparison"):
    """
    Plot multiple training histories on the same figure for comparison.
    histories_dict: dictionary { 'Network Name': keras_history_object }
    """
    plt.figure(figsize=(8,5))
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
