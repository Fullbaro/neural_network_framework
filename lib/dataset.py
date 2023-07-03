import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageClassificationDataset:

    def __init__(self):
        self.path: str
        self.labels: np.array
        self.X: np.array
        self.y: np.array

    def load(self, path, size, binary=False):
        self.path = path
        self.size = size
        self.labels = np.array(os.listdir(self.path))

        X_list = []
        y_list = []
        for label in self.labels:
            for file in os.listdir(os.path.join(self.path, label)):
                img = cv2.imread(os.path.join(self.path, label, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (size, size))

                X_list.append(img)
                y_list.append(label)

        self.X = np.array(X_list)
        self.y = np.array(y_list).astype('uint8')

        if binary:
            self.y = self.y.reshape(-1, 1) # Now classes are binary

    def preprocess(self, white_balance=False):
        for i in range(len(self.X)):
            img = self.X[i]

            if white_balance:
                img = cv2.equalizeHist(img)

            self.X[i] = img

    def augment(self):
        pass

    def balance(self):
        pass

    def shuffle(self):
        keys = np.array(range(self.X.shape[0]))
        np.random.shuffle(keys)
        self.X = self.X[keys]
        self.y = self.y[keys]

    def normalize(self):
        self.X = (self.X.astype(np.float32) - 127.5) / 127.5

    def reshape(self):
        self.X = self.X.reshape(self.X.shape[0], -1)

    def split(self, valid=0.2, test=0.1):
        dataset_size = self.X.shape[0]
        num_train = int(dataset_size * (1 - valid - test))
        num_valid = int(dataset_size * valid)
        num_test = int(dataset_size * test)

        self.X_train = self.X[:num_train]
        self.y_train = self.y[:num_train]
        self.X_valid = self.X[num_train:num_train + num_valid]
        self.y_valid = self.y[num_train:num_train + num_valid]
        self.X_test = self.X[num_train + num_valid:num_train + num_valid + num_test]
        self.y_test = self.y[num_train + num_valid:num_train + num_valid + num_test]

        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test


    def preview(self):
        print(f"Train batch contains {len(self.X_train):_} images")
        print(f"Validation batch contains {len(self.X_valid):_} images")
        print(f"Test batch contains {len(self.X_test):_} images")

        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0)

        for i, ax in enumerate(axs.flat):
            img = self.X[i].reshape((self.size, self.size))

            ax.imshow(img, cmap='gray')
            ax.set_title(self.y[i])
            ax.axis('off')

        plt.show()