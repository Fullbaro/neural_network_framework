import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

class ModelVisualizer:

    def visualize_train(self):
        y1 = self.history["train"]["acc"]
        y2 = self.history["valid"]["acc"]

        y3 = self.history["train"]["loss"]
        y4 = self.history["valid"]["loss"]

        y5 = self.history["train"]["lr"]

        y6 = self.history["train"]["dl"]
        y7 = self.history["train"]["rl"]

        x = list(range(1, len(y1) + 1))

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(x, y1, label='Training')
        axs[0, 0].plot(x, y2, label='Validation')
        axs[0, 0].set_ylabel('Accuracy %')
        axs[0, 0].set_xlabel('Epoch')

        axs[0, 1].plot(x, y3, label='Training')
        axs[0, 1].plot(x, y4, label='Validation')
        axs[0, 1].set_ylabel('Loss %')
        axs[0, 1].set_xlabel('Epoch')

        axs[1, 0].plot(x, y5)
        axs[1, 0].set_ylabel('Learning rate')
        axs[1, 0].set_xlabel('Epoch')

        axs[1, 1].plot(x, y6, label='Data loss')
        axs[1, 1].plot(x, y7, label='Regularization loss')
        axs[1, 1].set_ylabel('Loss %')
        axs[1, 1].set_xlabel('Epoch')

        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 1].legend()

        plt.show()

    def visualize_evaluation(self, X_test, y_test): # Ide akarok olyat mint a könyvben az animáció
        limit = len(X_test) if len(X_test) < 100 else 100
        view = TSNE(n_components=2).fit_transform(X_test[:limit])

        print(view[0:2])
        print(view.shape)



class ImageDatasetVisualizer:

    def visualize_dataset_preview(self):
        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0)

        for i, ax in enumerate(axs.flat):
            img = self.X[i].reshape((self.size, self.size))

            ax.imshow(img, cmap='gray')
            ax.set_title(self.y[i])
            ax.axis('off')
        plt.show()

    def visualize_dataset(self):
        limit = len(self.X_train) if len(self.X_train) < 2000 else 2000

        view = TSNE(n_components=2).fit_transform(self.X_train[:limit])
        plt.figure(figsize=(20,10))
        plt.scatter(view[:,0], view[:,1], c=self.y_train[:limit], alpha=0.5)
        plt.xlabel('t-SNE-1')
        plt.ylabel('t-SNE-2')
        plt.show()