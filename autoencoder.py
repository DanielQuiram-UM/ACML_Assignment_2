import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


class ConvAutoencoder:
    def __init__(self, input_shape=(32, 32, 3), version='full'):
        """
        version:
            'full'         -> original 9-layer network
            'fewer_layers' -> smaller network for testing
        """
        self.input_shape = input_shape
        self.version = version
        self.model = self.build_model()

    def build_model(self):
        input_img = layers.Input(shape=self.input_shape)
        x = input_img

        if self.version == 'full':
            # -----------------------------
            # Full network
            # -----------------------------
            # Encoder
            x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(x)
            encoded = layers.MaxPooling2D((2, 2))(x)

            # Decoder
            x = layers.UpSampling2D((2, 2))(encoded)
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(x)
            decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        elif self.version == 'fewer_layers':
            # -----------------------------
            # Smaller network for testing
            # -----------------------------
            # Encoder
            x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

            # Decoder
            x = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(encoded)
            x = layers.UpSampling2D((2, 2))(x)
            decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        else:
            raise ValueError(f"Unknown version '{self.version}' specified.")

        model = models.Model(input_img, decoded)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def train(self, x_train, x_val, epochs=10, batch_size=128):
        self.history = self.model.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, x_val)
        )
        return self.history

    def evaluate(self, x_test):
        test_loss = self.model.evaluate(x_test, x_test)
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
