import os
import torch
import scipy
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(dir_data: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads and processes data from the specified directory.
    If the data does not exist, generates a dummy dataset for demonstration.

    Args:
        dir_data (str): Path to the data directory.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the following elements:
            - x_train (torch.Tensor): A 2D tensor of shape (N, L) that represents the training input data, where:

                - **N**: The number of samples.
                - **L**: The length of each 1D sample.
            - y_train (torch.Tensor): A 2D tensor of shape (N, n_outputs) that represents the training output labels, where:

                - **n_outputs**: The dimension size of the ground truth values.
            - x_val (torch.Tensor): A 2D tensor of shape (N, L) that represents the validation input data.
            - y_val (torch.Tensor): A 2D tensor of shape (N, n_outputs) that represents the validation output labels.
            - x_test (torch.Tensor): A 2D tensor of shape (N, L) that represents the test input data.
            - y_test (torch.Tensor): A 2D tensor of shape (N, n_outputs) that represents the test output labels.
    """
    data_path = f'{dir_data}/data.mat'
    labels_path = f'{dir_data}/labels.mat'

    X = torch.tensor(X.get('X', X), dtype=torch.float32)
    Y = torch.tensor(Y.get('Y', Y), dtype=torch.float32)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

    return x_train, y_train, x_val, y_val, x_test, y_test


class Dataset():
    """
    Custom dataset class for handling input-output pairs.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Initializes the dataset with input and output data.

        Args:
            x (torch.Tensor): Input data tensor with size (N, L), where:

                - **N**: Number of samples.
                - **L**: Length of each 1D sample.
            y (torch.Tensor): Output labels tensor with size (N, n_outputs), where:

                - **n_outputs**: Number of output features.
        """
        self.x = x
        self.y = y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a data sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the following elements:

                - x (torch.Tensor): Input data tensor of size (L,), where:

                    - **L**: The length of each 1D sample.
                - y (torch.Tensor): Output label tensor of size (n_outputs,), where:
                    
                    - **n_outputs**: The number of output features.
        """
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.x)