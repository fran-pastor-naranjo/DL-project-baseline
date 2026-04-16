import torch
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def stop_count(self, loss: float, acc: float = None) -> bool:
        """
        Call this method at the end of each validation epoch to check if training should stop.
        
        Args:
            loss (float): The validation loss.
            acc (float, optional): Validation accuracy. Not used here natively but maintained for compatibility.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

        return self.early_stop


class ModelCheckpoint:
    """
    Saves the model weights if the given metric (e.g. validation loss) improves.
    """
    def __init__(self, filepath: str, epoch_start: int = 0, verbose: bool = False):
        """
        Args:
            filepath (str): The path to save the weights.
            epoch_start (int): Only consider saving the model after this epoch number.
            verbose (bool): Whether to print save events.
        """
        self.filepath = filepath
        self.epoch_start = epoch_start
        self.verbose = verbose
        self.best_loss = np.inf

    def checkpoint(self, loss: float, acc: float = None, epoch: int = 0, model: torch.nn.Module = None):
        """
        Check if the model should be saved at this epoch, and save it if needed.
        
        Args:
            loss (float): Target loss metric (e.g. sum of train and val loss, or val loss alone).
            acc (float, optional): Validation accuracy. Maintained for interface compatibility.
            epoch (int): Current epoch number.
            model (torch.nn.Module): The PyTorch model to save.
        """
        if epoch >= self.epoch_start:
            if loss < self.best_loss:
                if self.verbose:
                    print(f"Validation improved from {self.best_loss:.4f} to {loss:.4f}. Saving model to {self.filepath}")
                self.best_loss = loss
                torch.save(model.state_dict(), self.filepath)
