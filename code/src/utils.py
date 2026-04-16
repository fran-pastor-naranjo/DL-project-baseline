import os
import torch
import random
import numpy as np
from typing import Tuple
from sklearn.utils import class_weight
import pandas as pd
from time import sleep
import ast


def parse_list(x):
    """Safely evaluate list strings from pandas parsing."""
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def create_experiment(dir_results: str, ignored: list=[]) -> Tuple[str, str]:
    """
    Creates a new experiment directory by incrementing the highest existing numeric folder.
    
    Args:
        dir_results (str): The directory where experiment folders are stored.
        ignored (list, optional): List of folder names to ignore. Defaults to [].
    
    Returns:
        Tuple[str, str]: A tuple containing the following elements:

            - **experiment (str)**: The Experiment ID.
            - **file_name (str)**: The file path to the created folder.
    """
    existing_folders = [int(x) for x in os.listdir(dir_results + '/') if x.isdigit() and x not in ignored]
    experiment = str(np.max(existing_folders) + 1) if existing_folders else '1'
    file_name = os.path.join(dir_results, experiment)
    os.makedirs(file_name, exist_ok=True)
    return experiment, file_name


def get_class_weights(classes: torch.Tensor) -> torch.Tensor:
    """Computes class weights for imbalanced datasets.
    
    Args:
        classes (torch.Tensor): A 2D tensor of shape (N,C) that represents the vector of class labels in one-hot encoding, where:

            - **N**: Number of samples.
            - **C**: Number of classes.
    
    Returns:
        torch.Tensor: A 1D tensor of shape (C,) that represents the computed class weights.
    """
    class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                      classes=np.unique(np.argmax(classes, axis=1)), 
                                                      y=np.argmax(classes.detach().numpy(), axis=1))
    return torch.tensor(class_weights, dtype=torch.float32, 
                        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def set_seed(seed: int=42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch,
    and configures PyTorch to use deterministic operations.

    Args:
        seed (int, optional): Seed value to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Converts class labels to one-hot encoding.
    
    Args:
        y (numpy.ndarray): A 1D array of shape (N,) that represents the vector of class labels,  where:

            - **N**: Number of samples.
        num_classes (int): Number of classes.
    
    Returns:
        np.ndarray: One-hot encoded class labels. # DIMENSIONES??
    """
    return np.eye(num_classes, dtype='uint8')[y]


def is_valid_excel_file(filename: str) -> bool:
    """
    Check if an Excel file is valid and not corrupted.

    Args:
        filename (str): Path to the Excel file.

    Returns:
        bool: True if the file is a valid Excel file, False otherwise.
    """
    try:
        pd.ExcelFile(filename, engine='openpyxl')
        return True
    except Exception:
        return False


def write_dict_to_excel(data: pd.DataFrame, filename: str, sheet_name: str, max_retries: int=5, retry_wait: int=2) -> None:
    """
    Write a DataFrame to an Excel file, ensuring integrity and retrying on failure.

    Args:
        data (pd.DataFrame): The data to write to the Excel file.
        filename (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to write the data to.
        max_retries (int, optional): Maximum number of retry attempts if writing fails. Defaults to 5.
        retry_wait (int, optional): Seconds to wait between retries. Defaults to 2.

    Raises:
        RuntimeError: If writing fails after the maximum number of retries.
    """
    for attempt in range(max_retries):
        try:
            # Ensure file exists and is not corrupted
            if not os.path.exists(filename) or not is_valid_excel_file(filename):
                print(f"Recreating {filename} due to corruption or non-existence.")
                pd.DataFrame().to_excel(filename, engine='openpyxl')

            # Read existing sheets
            xls = pd.ExcelFile(filename, engine='openpyxl')
            sheet_dict = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

            if sheet_name in sheet_dict:
                sheet_dict[sheet_name] = pd.concat([sheet_dict[sheet_name], data], ignore_index=True)
            else:
                sheet_dict[sheet_name] = data

            # Write the updated data
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for sheet_name, df in sheet_dict.items():
                    df.to_excel(writer, index=False, sheet_name=sheet_name)

            print(f"Successfully wrote to {filename}.")
            return  # Exit function if successful

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            sleep(retry_wait)  # Wait before retrying

    raise RuntimeError(f"Failed to write to {filename} after {max_retries} attempts.")