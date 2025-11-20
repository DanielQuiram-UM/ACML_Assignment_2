from matplotlib import pyplot as plt

from color_coversion import color2ycrcb, ycrcb2color
from data_loader import DataLoader
from dataset import CIFAR10Dataset
from autoencoder import ConvAutoencoder
from training_visualization import plot_multiple_histories


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
    {'name': 'RGB output', 'version': 'full', 'skip':False, 'row':0, 'col':1},
    {'name': 'YCrCb output', 'version': 'full', 'skip':False, 'row':0, 'col':2},
    {'name': 'RGB output skip', 'version': 'full', 'skip':True, 'row':1, 'col':1},
    {'name': 'YCrCb output skip', 'version': 'full', 'skip':True, 'row':1, 'col':2},
]

all_histories = {}

im_indx = 12
fig, axes = plt.subplots(2, 3, figsize=(10, 5))
axes[0, 0].imshow(y_test_color[im_indx])
axes[0, 0].set_title('Color')
axes[1, 0].imshow(x_test_gray[im_indx], cmap='gray')
axes[1, 0].set_title('Gray scale')

for cfg in network_configs:
    print(f"\nTraining: {cfg['name']}")
    cae = ConvAutoencoder(input_shape=(32,32,1), version=cfg['version'], skip_connection=cfg['skip'])

    y_train, y_val, y_test = (y_train_color, y_val_color, y_test_color) if 'RGB' in cfg['name'] else \
                             (y_train_ycrcb, y_val_ycrcb, y_test_ycrcb)

    history = cae.train(x_train_gray, x_val_gray, y_train=y_train, y_val=y_val, epochs=10, batch_size=128)
    all_histories[cfg['name']] = history
    cae.evaluate(x_test_gray, y_test=y_test)

    y_pred = cae.model.predict(x_test_gray)
    if 'YCrCb' in cfg['name']:
        y_pred = ycrcb2color(y_pred)
    ax = axes[cfg['row'], cfg['col']]
    ax.imshow(y_pred[im_indx])
    ax.set_title(cfg['name'])
    ax.axis('off')
plt.tight_layout()

# Compare all networks on the same plot
plot_multiple_histories(all_histories)
