import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class ConvAutoencoder:
    def __init__(self, input_shape=(32, 32, 3), version=None, skip_connection=False, architecture=None):
        """
        Either specify:
        - version='full' or 'fewer_layers' (old-style hardcoded networks)
        OR
        - architecture=dict / OmegaConf (Hydra-configured network)
        """
        self.input_shape = input_shape
        self.version = version
        self.skip_connection = skip_connection
        self.architecture = architecture

        self.model = self.build_model()

    def build_model(self):
        x_in = layers.Input(shape=self.input_shape)
        x = x_in

        if self.architecture is not None:
            # -----------------------
            # Hydra-configured network
            # -----------------------
            skip_tensors = {}
            for layer_cfg in self.architecture.get('encoder', []):
                t = layer_cfg['type']
                if t == 'Conv2D':
                    x = layers.Conv2D(
                        filters=layer_cfg['filters'],
                        kernel_size=tuple(layer_cfg['kernel_size']),
                        activation=layer_cfg['activation'],
                        padding=layer_cfg['padding']
                    )(x)
                elif t == 'MaxPool':
                    x = layers.MaxPooling2D(
                        pool_size=tuple(layer_cfg['pool_size']),
                        padding=layer_cfg['padding']
                    )(x)
                if 'save' in layer_cfg:
                    skip_tensors[layer_cfg['save']] = x
            encoded = x

            for layer_cfg in self.architecture.get('decoder', []):
                t = layer_cfg['type']
                if t == 'Conv2D':
                    x = layers.Conv2D(
                        filters=layer_cfg['filters'],
                        kernel_size=tuple(layer_cfg['kernel_size']),
                        activation=layer_cfg['activation'],
                        padding=layer_cfg['padding']
                    )(x)
                elif t == 'UpSample':
                    x = layers.UpSampling2D(size=tuple(layer_cfg['size']))(x)
                elif t == 'Concat':
                    x = layers.Concatenate()([x, skip_tensors[layer_cfg['from_tensor']]])
            decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

        elif self.version is not None:
            # -----------------------
            # Old hard-coded networks
            # -----------------------
            if self.version == 'full':
                x1 = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
                x = layers.MaxPooling2D((2,2))(x1)
                x2 = layers.Conv2D(12, (3,3), activation='relu', padding='same')(x)
                encoded = layers.MaxPooling2D((2,2))(x2)

                x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
                x = layers.UpSampling2D((2,2))(x)
                if self.skip_connection:
                    x = layers.Concatenate()([x, x2])
                x = layers.Conv2D(12, (3,3), activation='relu', padding='same')(x)
                x = layers.UpSampling2D((2,2))(x)
                if self.skip_connection:
                    x = layers.Concatenate()([x, x1])
                decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

            elif self.version == 'fewer_layers':
                x1 = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
                encoded = layers.MaxPooling2D((2,2), padding='same')(x1)

                x = layers.Conv2D(12, (3,3), activation='relu', padding='same')(encoded)
                x = layers.UpSampling2D((2,2))(x)
                if self.skip_connection:
                    x = layers.Concatenate()([x, x1])
                decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

            else:
                raise ValueError(f"Unknown version '{self.version}' specified.")

        else:
            raise ValueError("Either 'version' or 'architecture' must be provided.")

        model = models.Model(x_in, decoded)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def train(self, x_train, x_val, y_train=None, y_val=None, epochs=10, batch_size=128):
        y_train = x_train if y_train is None else y_train
        y_val = x_val if y_val is None else y_val
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val)
        )
        return self.history

    def evaluate(self, x_test, y_test=None):
        y_test = x_test if y_test is None else y_test
        test_loss = self.model.evaluate(x_test, y_test)
        print(f"Test MSE ({self.version}):", test_loss)
        return test_loss

    def visualize_reconstructed_images(self, x_test, n=10):
        plt.figure(figsize=(10, 2 * n))
        decoded_imgs = self.model.predict(x_test[:n])

        for i in range(n):
            plt.subplot(n, 2, 2 * i + 1)
            plt.imshow(x_test[i])
            plt.axis("off")

            plt.subplot(n, 2, 2 * i + 2)
            plt.imshow(decoded_imgs[i])
            plt.axis("off")

        plt.tight_layout()
        plt.show(block=False)
