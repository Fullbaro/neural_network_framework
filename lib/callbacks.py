import numpy as np

class EarlyStoppingCallback:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.num_epochs_without_improvement = 0

    def on_epoch_end(self, epoch, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_epochs_without_improvement = 0
        else:
            self.num_epochs_without_improvement += 1

        if self.num_epochs_without_improvement >= self.patience:
            print(f"Training stopped due to lack of improvement in loss under {self.patience} epochs.")
            return True
        else:
            return False


# TODO: implement
class SaveBestCallback:

    def __init__(self):
        pass
