import tensorflow as tf

from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size, dim, n_channels, n_classes, data_path='', shuffle=True):
        self.data_path = data_path
        self.X = X
        self.y = y if y is not None else y
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __data_generation(self, X_list, y_list):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        if y is not None:
            for i, (img, label) in enumerate(zip(X_list, y_list)):
                X[i] = img
                y[i] = label

            return X, to_categorical(y, num_classes=self.n_classes)

        else:
            for i, img in enumerate(X_list):
                X[i] = img

            return X

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X_list = [self.X[k] for k in indexes]

        if self.y is not None:
            y_list = [self.y[k] for k in indexes]
            X, y = self.__data_generation(X_list, y_list)
            return X, y
        else:
            y_list = None
            X = self.__data_generation(X_list, y_list)
            return X