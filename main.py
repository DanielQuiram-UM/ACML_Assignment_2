from data_loader import DataLoader
from dataset import CIFAR10Dataset
from autoencoder import ConvAutoencoder
from training_visualization import plot_multiple_histories

# TODO: Seed-based reconstruction?

# Load data
data_dir = r"data\cifar-10-batches-py"
loader = DataLoader(data_dir)
x, y = loader.load_all_data()

dataset = CIFAR10Dataset(x, y)
dataset.normalize()
x_train, x_val, x_test = dataset.split()

# Network configurations
network_configs = [
    {'name': 'Baseline Network', 'version': 'full'},
    {'name': 'Fewer Layers Network', 'version': 'fewer_layers'},
]

all_histories = {}

for cfg in network_configs:
    print(f"\nTraining: {cfg['name']}")
    cae = ConvAutoencoder(input_shape=(32,32,3), version=cfg['version'])
    history = cae.train(x_train, x_val, epochs=100, batch_size=128)
    all_histories[cfg['name']] = history
    cae.evaluate(x_test)
    #cae.visualize_reconstructed_images(x_test)

# Compare all networks on the same plot
plot_multiple_histories(all_histories)
