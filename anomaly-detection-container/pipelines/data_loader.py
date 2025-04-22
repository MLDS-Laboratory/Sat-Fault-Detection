# pipelines/data_loader.py

import os
import numpy as np
import pandas as pd
import json
from typing import Union

def load_data(file_path: str) -> Union[pd.DataFrame, np.ndarray, dict]:
    """
    Loads data from a file. Supports CSV, npz, npy, and JSON formats.
    Returns:
        A DataFrame for CSV/JSON (if tabular), or a numpy array/dict.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".npy":
        return np.load(file_path, allow_pickle=True)
    elif ext == ".npz":
        return np.load(file_path, allow_pickle=True)
    elif ext == ".json":
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        """
        A simple DataLoader that supports pandas DataFrame, numpy arrays, and lists.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        if isinstance(data, pd.DataFrame):
            self.dataset = data.reset_index(drop=True)
            self.indices = list(self.dataset.index)
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            self.dataset = data
            self.indices = list(range(len(data)))
        else:
            raise ValueError("Unsupported data type for DataLoader.")

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index+self.batch_size]
        self.current_index += self.batch_size
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.iloc[batch_indices]
        elif isinstance(self.dataset, np.ndarray):
            return self.dataset[batch_indices]
        elif isinstance(self.dataset, list):
            return [self.dataset[i] for i in batch_indices]

def split_data(data, test_ratio=0.2):
    """
    Splits the data into training and testing sets.
    Works with pandas DataFrame and numpy array.
    """
    if isinstance(data, pd.DataFrame):
        data = data.sample(frac=1).reset_index(drop=True)  # Shuffle
        split_idx = int(len(data) * (1 - test_ratio))
        return data.iloc[:split_idx], data.iloc[split_idx:]
    elif isinstance(data, np.ndarray):
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_idx = int(len(data) * (1 - test_ratio))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        return data[train_idx], data[test_idx]
    else:
        raise ValueError("Unsupported data type for splitting.")
