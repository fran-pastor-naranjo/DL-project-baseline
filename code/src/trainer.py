import sys
import time
import torch
import pandas as pd
import torch.optim.lr_scheduler
from src.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict, List, Tuple

class Trainer:
    """
    A class to train and evaluate a machine learning model.

    This class handles the training process, including computing training and validation loss,
    evaluating the model, and handling callbacks such as model checkpointing and early stopping.
    """

    def __init__(self, epochs: int, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, callbacks: Tuple[ModelCheckpoint, EarlyStopping]=(None, None)) -> None:
        """
        Initializes the Trainer object with the provided parameters.

        Args:
            epochs (int): The number of epochs to train the model.
            criterion (torch.nn.Module): The loss function used for training.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.
            callbacks (List, optional): A List containing two callback objects, where:

                - **callbacks[0]**: ModelCheckpoint object for saving the model during training. Defaults to None.
                - **callbacks[1]**: EarlyStopping object for stopping training when validation loss does not improve. Defaults to None.
        """
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_checkpoint = callbacks[0]
        self.early_stop = callbacks[1]
        self.scheduler = scheduler

    def train(self, train_dl: torch.utils.data.DataLoader, val_dl: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> Dict[str, List[float]]:
        """
        Trains the model for the specified number of epochs while computing and logging the loss.

        Args:
            train_dl (torch.utils.data.DataLoader): The data loader for the training dataset.
            val_dl (torch.utils.data.DataLoader): The data loader for the validation dataset.
            model (torch.nn.Module): The model to train.
            device (torch.device): The device (CPU or GPU) to run the training on.

        Returns:
            H (Dict[str, List[float]]): A dictionary containing the training and validation loss history, where:
                - **'train_loss'**: A list of training loss values for each epoch.
                - **'val_loss'**: A list of validation loss values for each epoch.
        """
        train_loss_hist, val_loss_hist = list(), list()
        for epoch in range(self.epochs):

            sys.stdout.flush()
            epochStart = time.time()

            # TRAIN STAGE
            loss = self.forward_epoch(model, train_dl, device, train=True)
            train_loss_hist.append(loss)

            # VALIDATION STAGE
            loss = self.forward_epoch(model, val_dl, device, train=False)
            val_loss_hist.append(loss)

            # Show timing information for the epoch
            epochEnd = time.time()
            elapsed = (epochEnd - epochStart)

            # Show loss computed through the epoch
            print(f'Epoch {epoch+1}/{self.epochs} ({elapsed:.3} s): train loss: {train_loss_hist[-1]:.3} - val loss: {val_loss_hist[-1]:.3}')

            # Callbacks and scheduler execution
            if self.scheduler is not None:
                self.scheduler.step()

            if self.model_checkpoint is not None:
                self.model_checkpoint.checkpoint(loss=train_loss_hist[-1] + val_loss_hist[-1], acc=None, epoch=epoch, model=model)
            
            if self.early_stop is not None:
                stop = self.early_stop.stop_count(loss=val_loss_hist[-1], acc=None)
                if stop:
                    break

        history = {'train_loss':train_loss_hist, 'val_loss':val_loss_hist}
        return history

    def forward_batch(self, x: torch.Tensor, y: torch.Tensor, model: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the model on a single batch of data.

        Args:
            x (torch.Tensor): A 2D tensor of shape (N, L) that represents the input data, where:

                - **N**: Number of samples in the batch.
                - **L**: Length of each 1D sample.
            y (torch.Tensor): A 2D tensor of shape (N, n_outputs) that represents the ground truth labels, where:

                - **n_outputs**: Number of output features.
            model (torch.nn.Module): The model to evaluate.
            device (torch.device): The device (CPU or GPU) to run the evaluation on.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The loss tensor (scalar) representing the computed loss value.
                - The predictions tensor with shape (N, n_outputs)
        """                                    
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = self.criterion(pred, y.float())

        return loss, pred
    
    def forward_epoch(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, train: bool=True) -> float:
        """
        Forward pass through the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train or evaluate.
            dataloader (torch.utils.data.DataLoader): The data loader for the dataset.
            device (torch.device): The device (CPU or GPU) to run the training or evaluation on.
            train (bool, optional): Whether to train the model or just evaluate. Defaults to True.

        Returns:
            float: The average loss for the epoch.
        """
        epoch_loss = 0
        if train:
            model.train()
        else:
            model.eval()

        for i, (x, y) in enumerate(dataloader):
            if train:
                self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(train):
                loss, _ = self.forward_batch(x, y, model, device)
                if train:
                    loss.backward()
                    self.optimizer.step()

            # Save metrics
            epoch_loss += loss.cpu().detach().numpy()
        
        return epoch_loss / len(dataloader)
    
    def test(self, test_dl: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> pd.DataFrame:
        """
        Evaluates the model on the test dataset and saves the predictions and ground truth to an Excel file.

        Args:
            test_dl (torch.utils.data.DataLoader): The data loader for the test dataset.
            model (torch.nn.Module): The model to evaluate.
            device (torch.device): The device (CPU or GPU) to run the evaluation on.
        Returns:
            pd.DataFrame: A DataFrame containing the predictions and ground truth for each sample in the test dataset.
        """
        predictions, targets = list(), list()

        model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                _, pred = self.forward_batch(x, y, model, device)

                predictions.append(pred.cpu().detach().numpy())
                targets.append(y.cpu().detach().numpy())

        for i in range(len(predictions)):
            # Generate the dataframe
            df = pd.DataFrame([{
                                'prediction': predictions[i],
                                'ground_truth': targets[i]
                                }])
        return df
    