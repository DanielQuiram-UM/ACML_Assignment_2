from matplotlib import pyplot as plt

from color_coversion import color2ycrcb
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
y_train_color, y_val_color, y_test_color = dataset.split()

y_train_ycrcb, x_train_gray  = color2ycrcb(y_train_color)
y_val_ycrcb, x_val_gray  = color2ycrcb(y_val_color)
y_test_ycrcb, x_test_gray  = color2ycrcb(y_test_color)

# Network configurations
network_configs = [
    {'name': 'RGB output', 'version': 'full', 'skip':False},
    {'name': 'YCrCb output', 'version': 'full', 'skip':False},
    {'name': 'RGB output skip', 'version': 'full', 'skip':True},
    {'name': 'YCrCb output skip', 'version': 'full', 'skip':True},
]

all_histories = {}

for cfg in network_configs:
    print(f"\nTraining: {cfg['name']}")
    cae = ConvAutoencoder(input_shape=(32,32,1), version=cfg['version'], skip_connection=cfg['skip'])

    y_train, y_val, y_test = (y_train_color, y_val_color, y_test_color) if cfg['name'] == 'RGB output' else \
                             (y_train_ycrcb, y_val_ycrcb, y_test_ycrcb)

    history = cae.train(x_train_gray, x_val_gray, y_train=y_train, y_val=y_val, epochs=100, batch_size=128)
    all_histories[cfg['name']] = history
    cae.evaluate(x_test_gray, y_test=y_test)
    #cae.visualize_reconstructed_images(x_test)

# Compare all networks on the same plot
plot_multiple_histories(all_histories)
