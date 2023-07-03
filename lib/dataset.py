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

    def load(self, path, size):
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

    def preprocess(self, grayscale=False, white_balance=False):
        for i in range(len(self.X)):
            img = self.X[i]

            if white_balance and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                avg_a = np.average(img[:, :, 1])
                avg_b = np.average(img[:, :, 2])
                img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)
                img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)
                img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

            if grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        # if len(self.labels) == 2:
        #     self.y = self.y.reshape(-1, 1)

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