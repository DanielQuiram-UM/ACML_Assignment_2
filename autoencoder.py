import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


class ConvAutoencoder:
    def __init__(self, input_shape=(32, 32, 3), version='full', skip_connection=False):
        """
        version:
            'full'         -> original 9-layer network
            'fewer_layers' -> smaller network for testing
        """
        self.input_shape = input_shape
        self.version = version
        self.skip_connection = skip_connection

        self.model = self.build_model()

    def build_model(self):
        input_img = layers.Input(shape=self.input_shape)
        x = input_img

        if self.version == 'full':
            # -----------------------------
            # Full network
            # -----------------------------
            # Encoder
            x1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x1)
            x2 = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(x)
            encoded = layers.MaxPooling2D((2, 2))(x2)

            # Decoder
            x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
            x = layers.UpSampling2D((2, 2))(x)
            if self.skip_connection:
                x = layers.Concatenate()([x, x2])
            x = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(x)
            x = layers.UpSampling2D((2, 2))(x)
            if self.skip_connection:
                x = layers.Concatenate()([x, x1])
            decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        elif self.version == 'fewer_layers':
            # -----------------------------
            # Smaller network for testing
            # -----------------------------
            # Encoder
            x1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            encoded = layers.MaxPooling2D((2, 2), padding='same')(x1)

            # Decoder
            x = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(encoded)
            x = layers.UpSampling2D((2, 2))(x)
            if self.skip_connection:
                x = layers.Concatenate()([x, x1])
            decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        else:
            raise ValueError(f"Unknown version '{self.version}' specified.")

        model = models.Model(input_img, decoded)
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
