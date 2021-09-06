"""Module for experiments' dataset methods."""
import os
import random
from typing import Tuple

import numpy as np
from pims import ImageSequence
from torch.utils.data import Dataset, Subset


class CPDDatasets:
    """Class for experiments' datasets."""

    def __init__(self, experiments_name: str) -> None:
        """Initialize class.

        :param experiments_name: type of experiments (only mnist available now!)
        """
        super().__init__()

        if experiments_name in [
            "synthetic",
            "human_activity",
            "mnist",
            "explosion",
            "oops",
        ]:
            self.experiments_name = experiments_name
        else:
            raise ValueError("Wrong experiment_name {}.".format(experiments_name))

    def get_dataset_(self) -> Tuple[Dataset, Dataset]:
        """Load experiments' dataset. Only MNIST available."""
        train_dataset = None
        test_dataset = None
        if self.experiments_name == "mnist":
            path_to_data = "data/mnist/"
            dataset = MNISTSequenceDataset(path_to_data=path_to_data, type_seq="all")
            train_dataset, test_dataset = CPDDatasets.train_test_split_(
                dataset, test_size=0.3, shuffle=True
            )
        return train_dataset, test_dataset

    @staticmethod
    def train_test_split_(
        dataset: Dataset, test_size: float = 0.3, shuffle: bool = True
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset on train and test.

        :param dataset: dataset for splitting
        :param test_size: size of test data
        :param shuffle: if True, shuffle data
        :return: tuple of
            - train dataset
            - test dataset
        """
        len_dataset = len(dataset)
        idx = np.arange(len_dataset)

        if shuffle:
            train_idx = random.sample(list(idx), int((1 - test_size) * len_dataset))
        else:
            train_idx = idx[: -int(test_size * len_dataset)]
        test_idx = np.setdiff1d(idx, train_idx)

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        return train_set, test_set


class MNISTSequenceDataset(Dataset):
    """Class for Dataset consists of sequences of MNIST images."""

    def __init__(self, path_to_data: str, type_seq: str = "all") -> None:
        """Initialize datasets' parameters.

        :param path_to_data: path to folders with MNIST sequences
        :param type_seq: type of data for loading (only normal, only anomaly, all)
        """
        super().__init__()

        # set paths to data
        self.path_to_data = path_to_data
        self.path_to_normal = os.path.join(path_to_data, "normal/")
        self.path_with_change = os.path.join(path_to_data, "with_change/")

        self.normal_seq_paths = [
            os.path.join(self.path_to_normal, x)
            for x in os.listdir(self.path_to_normal)
        ]
        self.with_change_seq_paths = [
            os.path.join(self.path_with_change, x)
            for x in os.listdir(self.path_with_change)
        ]

        # load all sequences, only normal or only sequences with changes
        if type_seq == "all":
            self.sample_paths = self.normal_seq_paths + self.with_change_seq_paths
        elif type_seq == "normal":
            self.sample_paths = self.normal_seq_paths
        elif type_seq == "only_changes":
            self.sample_paths = self.with_change_seq_paths
        else:
            raise ValueError(
                'Unknown label type "{}". Please, choose one of {{all, normal, only_changes}}.'.format(
                    type_seq
                )
            )

    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.normal_seq_paths) + len(self.with_change_seq_paths)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """Get one images' sequence and corresponding labels from dataset.

        :param idx: index of element in dataset
        :return: tuple of
             - sequence of images
             - sequence of labels
        """
        # read sequences of images
        path_img = self.sample_paths[idx]
        seq_images = ImageSequence(os.path.join(path_img, "*_*.png"))
        seq_images = np.transpose(seq_images, (0, 3, 1, 2))[:, 0, :, :].astype(float)
        seq_labels = sorted(os.listdir(path_img), key=lambda x: int(x.split("_")[0]))

        # get corresponding labels
        seq_labels = [int(x.split(".png")[0].split("_")[1]) for x in seq_labels]
        seq_labels = (np.array(seq_labels) != seq_labels[0]).astype(int)

        return seq_images, seq_labels

    @staticmethod
    def convert_to_gray_(frame: np.array) -> np.array:
        """Convert PIMS' images to gray scale. In MNIST case, all channels are equals.

        :param frame: image
        :return: image in gray scale
        """
        return frame[:, :, 0]
