import os
import pickle
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_batch(self, batch_filename):

        with open(batch_filename, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            data = batch['data']
            labels = batch['labels']
            data = data.reshape((len(data), 3, 32, 32))
            data = np.transpose(data, (0, 2, 3, 1))
        return data, np.array(labels)

    def load_all_data(self):

        # Load training batches (all five)
        x_train_list, y_train_list = [], []
        for i in range(1, 6):
            batch_file = os.path.join(self.data_dir, f"data_batch_{i}")
            data_batch, labels_batch = self.load_batch(batch_file)
            x_train_list.append(data_batch)
            y_train_list.append(labels_batch)

        x_train_full = np.concatenate(x_train_list)
        y_train_full = np.concatenate(y_train_list)

        # Load the official CIFAR10 test set and combine it with the data
        x_test, y_test = self.load_batch(os.path.join(self.data_dir, "test_batch"))

        x = np.concatenate([x_train_full, x_test])
        y = np.concatenate([y_train_full, y_test])

        return x, y
