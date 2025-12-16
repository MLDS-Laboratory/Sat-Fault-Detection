import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

from pipelines.esa_dataloader import ESAMissionDataLoader

class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        #for 10% few-shot learning
        # data = data[:int(data.shape[0] * 0.05)]

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        # len_train = int(data.shape[0] * 0.8)
        # self.train = data[:len_train]
        # self.val = data[len_train:]

        self.train = data
        self.val = self.test


        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print('val:', self.val.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")

        # for 10% few-shot learning
        # data = data[:int(data.shape[0] * 0.05)]

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        # len_train = int(data.shape[0] * 0.8)
        # self.train = data[:len_train]
        # self.val = data[len_train:]

        self.train = data
        self.val = self.test

        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")

        # for 10% few-shot learning
        # data = data[:int(data.shape[0] * 0.05)]


        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        # len_train = int(data.shape[0] * 0.8)
        # self.train = data[:len_train]
        # self.val = data[len_train:]

        self.train = data
        self.val = self.test

        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")

        # for 10% few-shot learning
        # data = data[:int(data.shape[0] * 0.05)]

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)

        len_train = int(data.shape[0] * 0.8)
        self.train = data
        self.val = data[len_train:]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]

        # for 10% few-shot learning
        # train_data = train_data[:int(train_data.shape[0] * 0.05)]


        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        len_train = int(train_data.shape[0] * 0.8)
        self.train = train_data
        self.test = test_data
        self.val = train_data[len_train:]

        # self.train = train_data
        # self.test = test_data
        # self.val = self.test

        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1


    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class NIPSSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = pd.read_csv(os.path.join(root_path, 'nips.csv')).values
        train_data = data[:20000, :1]
        test_data = data[30000:, :2]
        labels = test_data[:, -1:]
        test_data = test_data[:, :1]
        val_data = data[20000:30000, :1]

        # for 10% few-shot learning
        # train_data = train_data[:int(train_data.shape[0] * 0.05)]



        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        val_data = self.scaler.transform(val_data)

        # len_train = int(train_data.shape[0] * 0.8)
        self.train = train_data
        self.test = test_data
        self.val = val_data
        # self.val = train_data[len_train:]

        # self.train = train_data
        # self.test = test_data
        # self.val = self.test

        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1


    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class ESASegLoader(Dataset):
    """
    Adapter that takes your ESAMissionDataLoader (segment-based)
    and turns it into a GDFormer-style sliding-window dataset.

    It treats each channel as a 1-D time series (input_c = 1)
    and concatenates all segments into long 1-D sequences for
    train/test, with per-sample labels for the test sequence.

    Modes:
        - train: sliding windows over nominal training data
        - val:   sliding windows over the same distribution as test
        - test:  sliding windows over test data (with labels aligned)
        - thre:  non-overlapping windows over test data (for thresholding)
    """

    def __init__(
        self,
        mission_dir: str,
        win_size: int,
        step: int = 1,
        mode: str = "train",
        nominal_segment_len: int | None = None,
        train_ratio: float = 0.8,
        random_state: int = 42,
        use_only_nominal_train: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # -------------------------------
        # 1) Load ESA segments
        # -------------------------------
        esa_loader = ESAMissionDataLoader(
            mission_dir=mission_dir,
            nominal_segment_len=nominal_segment_len,
            train_ratio=train_ratio,
            random_state=random_state,
        )
        train_segments, test_segments = esa_loader.get_train_test_segments()

        # -------------------------------
        # 2) Build long 1-D sequences
        # -------------------------------

        # Training data: by default, *only* nominal segments for unsupervised training
        if use_only_nominal_train:
            train_nominal = [s["ts"] for s in train_segments if s["label"] == 0]
        else:
            train_nominal = [s["ts"] for s in train_segments]

        if len(train_nominal) == 0:
            # Fallback: if for some reason there are no nominal segments
            train_nominal = [s["ts"] for s in train_segments]

        train_data = np.concatenate(train_nominal, axis=0).astype(np.float32)

        # Test data: all segments, keep per-sample labels (segment label repeated)
        test_series = []
        test_labels = []
        for s in test_segments:
            ts = np.asarray(s["ts"], dtype=np.float32)
            lbl = int(s["label"])
            test_series.append(ts)
            test_labels.append(np.full(len(ts), lbl, dtype=np.int64))

        if len(test_series) == 0:
            raise RuntimeError("No test segments found in ESA loader.")

        test_data = np.concatenate(test_series, axis=0).astype(np.float32)
        test_labels = np.concatenate(test_labels, axis=0).astype(np.int64)

        # Shape to (N, 1) so enc_in = 1
        train_data = train_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)
        test_labels = test_labels.reshape(-1, 1)

        # -------------------------------
        # 3) Standardize on train
        # -------------------------------
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        # Following MSL/SMAP convention: val uses test distribution
        self.train = train_data
        self.val = test_data
        self.test = test_data
        self.test_labels = test_labels

        print("ESA train:", self.train.shape)
        print("ESA val:  ", self.val.shape)
        print("ESA test: ", self.test.shape)

    # ----------------------------------------------------
    # Sliding window indexing – same semantics as GDFormer
    # ----------------------------------------------------
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            # 'thre' mode: non-overlapping windows over the test set
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = self.train[index:index + self.win_size]
            # Training labels are unused in GDFormer; just give something of right shape
            y = self.test_labels[0:self.win_size]
        elif self.mode == "val":
            x = self.val[index:index + self.win_size]
            y = self.test_labels[0:self.win_size]
        elif self.mode == "test":
            x = self.test[index:index + self.win_size]
            y = self.test_labels[index:index + self.win_size]
        else:
            # 'thre' – non-overlapping windows over the test set
            base = (index // self.step) * self.win_size
            x = self.test[base:base + self.win_size]
            y = self.test_labels[base:base + self.win_size]

        return np.float32(x), np.float32(y)




def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT'):
        dataset = SWATSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'NIPS'):
        dataset = NIPSSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ESA'):
        dataset = ESASegLoader(
            mission_dir=data_path,
            win_size=win_size,
            step=1, 
            mode=mode,
            nominal_segment_len=None,
            train_ratio=0.8,
            random_state=42,
            use_only_nominal_train=True,
        )
        

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
