from sklearn.model_selection import train_test_split

class CIFAR10Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def normalize(self):
        self.x = self.x.astype('float32') / 255.0
        return self

    def split(self, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
        """
        Split the data into 80% Train, 10% Validation and 10% Test
        """
        x_train, x_temp = train_test_split(self.x, test_size=(1-train_size), random_state=random_state)
        val_ratio = val_size / (val_size + test_size)
        x_val, x_test = train_test_split(x_temp, test_size=(1-val_ratio), random_state=random_state)
        return x_train, x_val, x_test
