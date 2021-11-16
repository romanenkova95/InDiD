"""Module for experiments' dataset methods."""
import os
import random
import torch
from typing import Tuple

import numpy as np
import pandas as pd
from pims import ImageSequence
from torch.utils.data import Dataset, Subset
from torch.distributions.multivariate_normal import MultivariateNormal


class CPDDatasets:
    """Class for experiments' datasets."""

    def __init__(self, experiments_name: str, model_type='seq2seq') -> None:
        """Initialize class.

        :param experiments_name: type of experiments (only mnist available now!)
        """
        super().__init__()

        # TODO make 
        if experiments_name in [
            "human_activity",
            "mnist",
            "explosion",
            "oops",
        ]:
            self.experiments_name = experiments_name
        elif experiments_name.startswith('synthetic'):
            # format is "synthetic_nD"
            self.D = int(experiments_name.split('_')[1].split('D')[0])
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
            
        elif self.experiments_name == "oops":
            path_to_train_data = "data/oops/train_data"
            path_to_val_data = "data/oops/val_data"
            #train_dataset = OOPSSequenceDataset.get_subset(path_to_data=path_to_train_data,
            #                                               subset_size=1000)
            #test_dataset = OOPSSequenceDataset.get_subset(path_to_data=path_to_val_data,
            #                                             subset_size=1000)
            train_dataset = OOPSSequenceDataset(path_to_data=path_to_train_data)
            test_dataset = OOPSSequenceDataset(path_to_data=path_to_val_data)

        elif self.experiments_name.startswith("synthetic"):
            dataset = SyntheticNormalDataset(seq_len=128, num=1000, D=self.D, random_seed=123)
            train_dataset, test_dataset = CPDDatasets.train_test_split_(
                dataset, test_size=0.3, shuffle=True
            )
            
        elif self.experiments_name == "human_activity":
            path_to_data = "data/human_activity/"
            train_dataset = HumanActivityDataset(path_to_data=path_to_data, seq_len=20, train_flag=True)
            test_dataset = HumanActivityDataset(path_to_data=path_to_data, seq_len=20, train_flag=False)
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
    
    @staticmethod
    def get_subset_(
        dataset: Dataset, subset_size: int, shuffle: bool = True
    ) -> Dataset:
        len_dataset = len(dataset)
        idx = np.arange(len_dataset)

        if shuffle:
            idx = random.sample(min(subset_size, len_dataset))
        else:
            idx = idx[: subset_size]
        subset = Subset(dataset, idx)
        return subset  
    


class MNISTSequenceDataset(Dataset):
    """Class for Dataset consists of sequences of MNIST images."""

    def __init__(self, path_to_data: str, 
                 type_seq: str = "all", baseline: bool = False) -> None:
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
        
        
        self.baseline = baseline

    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        if self.baseline:
            self.baseline = False
            tmp_imgs, tmp_labels = self.__getitem__(0)
            self.baseline = True       
            self.seq_len = len(tmp_labels)
            len_dataset = len(self.sample_paths) * self.seq_len
        else:
            len_dataset = len(self.sample_paths)
        return len_dataset

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """Get one images' sequence and corresponding labels from dataset.

        :param idx: index of element in dataset
        :return: tuple of
             - sequence of images
             - sequence of labels
        """
        # read sequences of images
        if self.baseline:
            dataset_idx = idx // self.seq_len
        else:
            dataset_idx = idx
        
        path_img = self.sample_paths[dataset_idx]
        seq_images = ImageSequence(os.path.join(path_img, "*_*.png"))
        seq_images = np.transpose(seq_images, (0, 3, 1, 2))[:, 0, :, :].astype(float)
        seq_labels = sorted(os.listdir(path_img), key=lambda x: int(x.split("_")[0]))

        # get corresponding labels
        seq_labels = [int(x.split(".png")[0].split("_")[1]) for x in seq_labels]
        seq_labels = (np.array(seq_labels) != seq_labels[0]).astype(int)

        
        if self.baseline:
            img_idx = idx % self.seq_len
            seq_images = seq_images[img_idx] 
            seq_labels = seq_labels[img_idx]
        
        return seq_images, seq_labels

    @staticmethod
    def convert_to_gray_(frame: np.array) -> np.array:
        """Convert PIMS' images to gray scale. In MNIST case, all channels equals.

        :param frame: image
        :return: image in gray scale
        """
        return frame[:, :, 0]

class OOPSSequenceDataset(Dataset):
    """Class for Dataset consists of sequences of OOPS images."""

    def __init__(self, path_to_data: str) -> None:
        """Initialize datasets' parameters.

        :param path_to_data: path to folders with OOPS sequences
        """
        super().__init__()

        # set paths to data
        self.path_to_data = path_to_data

        self.sample_paths = [
            os.path.join(self.path_to_data, x)
            for x in os.listdir(self.path_to_data)
        ]

    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.sample_paths)

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
        seq_images = np.transpose(seq_images, (0, 3, 1, 2)).astype(float)
        # TODO: fix
        seq_images = np.transpose(seq_images, (1, 0, 2, 3)).astype(np.float64)

        seq_labels = sorted(os.listdir(path_img), key=lambda x: int(x.split("_")[0]))

        # get corresponding labels
        seq_labels = [int(x.split(".png")[0].split("_")[1]) for x in seq_labels]
        seq_labels = (np.array(seq_labels) != seq_labels[0]).astype(int)

        return seq_images, seq_labels
    
    @staticmethod
    def get_subset(path_to_data: str, subset_size: int, random_seed=123):
        """Get subset of dataset.

        :param path_to_data: path to data
        :param subset_size: size of subset
        :param random_seed: fix seed for reproduction
        
        :return: subset of dataset with corresponded size
        """
        
        dataset = OOPSSequenceDataset(path_to_data)
        
        np.random.seed(random_seed)
        idx = np.random.randint(0, len(dataset), size=subset_size)
        
        return Subset(dataset, idx)  
    
    
class SyntheticNormalDataset(Dataset):

    def __init__(self, seq_len: int, num: int, D=1, random_seed=123):

        super().__init__()

        self.data, self.labels = SyntheticNormalDataset.generate_synthetic_nD_data(seq_len, num, 
                                                                                   D=D, random_seed=123)
    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:

        return self.data[idx], self.labels[idx]

    @staticmethod
    def generate_synthetic_nD_data(seq_len, num, D=1, random_seed=123, multi_dist=False):
        torch.manual_seed(random_seed)

        idxs_changes = torch.randint(1, seq_len, (num // 2, ))
        
        data = []
        labels = []

        for idx in idxs_changes:
            mu = torch.randint(1, 100, (2, ))
            
            while mu[0] == mu[1]:
                mu = torch.randint(1, 100, (2, ))
            
            if not multi_dist:
                mu[0] = 1 
           
            m = MultivariateNormal(mu[0] * torch.ones(D), torch.eye(D))    
            x1 = []
            for _ in range(seq_len):
                x1_ = m.sample()
                x1.append(x1_)
            x1 = torch.stack(x1)

            m = MultivariateNormal(mu[1] * torch.ones(D), torch.eye(D))    
            x2 = []
            for _ in range(seq_len):
                x2_ = m.sample()
                x2.append(x2_)
            x2 = torch.stack(x2)            
                       
            x = torch.cat([x1[:idx], x2[idx:]])
            label = torch.cat([torch.zeros(idx), torch.ones(seq_len-idx)])
            data.append(x)
            labels.append(label)

        for idx in range(0, num - len(idxs_changes)):            
            m = MultivariateNormal(torch.ones(D), torch.eye(D))    
            x = []
            for _ in range(seq_len):
                x_ = m.sample()
                x.append(x_)
            x = torch.stack(x)            
            label = torch.zeros(seq_len)
            
            data.append(x)
            labels.append(label)
        return data, labels
    

#TODO Reformat
class HumanActivityDataset(Dataset):

    def __init__(self, path_to_data, seq_len, train_flag=True):

        super().__init__()

        data = self.load_data(path_to_data, train_flag=train_flag)
        normal_data, normal_labels = self.generate_normal_data(data, seq_len)
        anomaly_data, anomaly_labels = self.generate_anomaly_data(data, seq_len)

        self.features = normal_data + anomaly_data
        self.labels = normal_labels + anomaly_labels
        
    def load_data(self, path, train_flag=True):

        if train_flag:
            type_ = "train"
        else:
            type_ = "test"

        data_path = path + "/T" + type_[1:] + "/X_" + type_ + ".txt"
        labels_path = path + "/T" + type_[1:] + "/y_" + type_ + ".txt"
        subjects_path = path + "/T" + type_[1:] + "/subject_id_" + type_ + ".txt"

        data = pd.read_csv(data_path, sep=" ", header=None)
        names = pd.read_csv(path + '/features.txt', header=None)        
        data.columns = [x.replace(' ', '') for x in names[0].values]

        labels = pd.read_csv(labels_path, sep=" ", header=None)
        subjects = pd.read_csv(subjects_path, sep=" ", header=None)

        data["subject"] = subjects
        data["labels"] = labels

        return data    

    
    def generate_normal_data(self, data, seq_len=20):
        slices = []
        labels = []

        for sub in data['subject'].unique():

            tmp = data[data.subject == sub]
            # labels 7 - 12 characterize the change points
            tmp = tmp[~tmp['labels'].isin([7, 8, 9, 10, 11, 12])]
            normal_ends_idxs = np.where(np.diff(tmp['labels'].values) != 0)[0]

            start_idx = 0

            for i in range(0, len(normal_ends_idxs) - 1):
                # get data before change 
                end_idx = normal_ends_idxs[i] + 1
                slice_data = tmp.iloc[start_idx: end_idx]

                if len(slice_data) > seq_len:
                    for i in range(0, len(slice_data) - seq_len):
                        seq = slice_data[i: i + seq_len]
                        slices.append(seq)    
                        labels.append(np.zeros(len(seq)))
                start_idx = end_idx

        return slices, labels

    
    def generate_anomaly_data(self, data, seq_len=20):
        slices = []
        labels = []

        for sub in data['subject'].unique():

            tmp = data[data.subject == sub]
            # labels 7 - 12 characterize the change points
            tmp_change_only = tmp[tmp.labels.isin([7, 8, 9, 10, 11, 12])]
            change_idxs = np.where(np.diff(tmp_change_only['labels'].values) != 0)[0]
            change_idxs = [tmp_change_only.index[0]] + list(tmp_change_only.index[change_idxs + 1])
            change_idxs = [-1] + change_idxs

            for i in range(1, len(change_idxs) - 1):
                curr_change = change_idxs[i]            
                start_idx = max(change_idxs[i - 1] + 1, curr_change - seq_len)
                end_idx = min(change_idxs[i + 1] - 1, curr_change + seq_len)

                slice_data = tmp.loc[start_idx: end_idx]

                curr_change = list(slice_data.index).index(curr_change)

                slice_labels = np.zeros(len(slice_data))   
                slice_labels[curr_change:] = np.ones(len(slice_data) - curr_change) 

                if len(slice_data) > seq_len:
                    for i in range(0, len(slice_data) - seq_len):
                        seq = slice_data.iloc[i: i + seq_len]
                        label = slice_labels[i : i + seq_len]
                        slices.append(seq)    
                        labels.append(label)
        return slices, labels
    
    
    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        sel_features = ['tBodyAcc-Mean-1', 'tBodyAcc-Mean-2', 'tBodyAcc-Mean-3', 
                        'tGravityAcc-Mean-1', 'tGravityAcc-Mean-2', 'tGravityAcc-Mean-3',
                        'tBodyAccJerk-Mean-1', 'tBodyAccJerk-Mean-2', 'tBodyAccJerk-Mean-3',
                        'tBodyGyro-Mean-1', 'tBodyGyro-Mean-2', 'tBodyGyro-Mean-3', 
                        'tBodyGyroJerk-Mean-1', 'tBodyGyroJerk-Mean-2', 'tBodyGyroJerk-Mean-3',
                        'tBodyAccMag-Mean-1', 'tGravityAccMag-Mean-1', 'tBodyAccJerkMag-Mean-1',
                        'tBodyGyroMag-Mean-1', 'tBodyGyroJerkMag-Mean-1', 'fBodyAcc-Mean-1',
                        'fBodyAcc-Mean-2', 'fBodyAcc-Mean-3', 'fBodyGyro-Mean-1', 'fBodyGyro-Mean-2', 
                        'fBodyGyro-Mean-3', 'fBodyAccMag-Mean-1', 'fBodyAccJerkMag-Mean-1', 
                        'fBodyGyroMag-Mean-1', 'fBodyGyroJerkMag-Mean-1']        
        
        
        return self.features[idx][sel_features].iloc[:, :-2].values, self.labels[idx]    
    
    
class BaselineDataset(Dataset):
    def __init__(self, cpd_dataset, baseline_type='simple', subseq_len=None):
        self.baseline_type = baseline_type
        
        self.cpd_dataset = cpd_dataset
        self.N = len(cpd_dataset)
        self.T = len(cpd_dataset.__getitem__(0)[0])
        
        self.subseq_len = subseq_len
        
        if (self.baseline_type == 'weak_labels') and (self.subseq_len is None):
            raise ValueError('Please, set subsequence length.')
        
    def __len__(self):
        if self.baseline_type == 'simple':
            return self.N * self.T
        elif (self.baseline_type == 'weak_labels'):
            return self.N * (self.T - self.subseq_len + 1)
        else:
            raise ValueError("Wrong type of baseline.")
        
    def __getitem__(self, idx):
        if self.baseline_type == 'simple':
            global_idx = idx // self.T
            local_idx = idx % self.T
            images = self.cpd_dataset[global_idx][0][local_idx]
            labels = self.cpd_dataset[global_idx][1][local_idx] 
            
        elif self.baseline_type == 'weak_labels':
            global_idx = idx // (self.T - self.subseq_len + 1)
            local_idx = idx %  (self.T - self.subseq_len + 1)
            print(global_idx)
            print(local_idx)            
            images = self.cpd_dataset[global_idx][0][local_idx: local_idx + self.subseq_len]
            print(self.cpd_dataset[global_idx][1])
            labels = max(self.cpd_dataset[global_idx][1][local_idx: local_idx + self.subseq_len])
            
        return images, labels    
