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
#SEQ_LEN = 64 * 8

class CPDDatasets:
    """Class for experiments' datasets."""

    def __init__(self, experiments_name: str =  "mnist128" ) -> None:
        """Initialize class.

        :param experiments_name: type of experiments (only mnist available now!)
        """
        super().__init__()

        # TODO make 
        if experiments_name in [
            "mnist64", "mnist128", "mnist256", "mnist512", "mnist768", "mnist1024", "mnist", 
            "1changes", "2changes", "4changes", "6changes", "8changes", "10changes",
            "topic_segmentation"
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
        if self.experiments_name[:5] == "mnist":
            if self.experiments_name[5:] == '':
                self.experiments_name += '64'
            path_to_data = "/mnt/data/eromanenkova/Intern_CPD/Anna/data/128/"

            dataset = MNISTSequenceDataset(path_to_data=path_to_data, type_seq="all")
            train_dataset, test_dataset = CPDDatasets.train_test_split_(
                dataset, test_size=0.3, shuffle=True
            )
        if self.experiments_name[-7:] == "changes":
            path_to_data = "/mnt/data/eromanenkova/Intern_CPD/Anna" + "data/"+ self.experiments_name + "/"
     
            dataset = MNISTSequenceDataset(path_to_data=path_to_data, type_seq="all")
            train_dataset, test_dataset = CPDDatasets.train_test_split_(
                dataset, test_size=0.3, shuffle=True
            )

        #----------------------------------------------------------------------------------------#
        ##########################################################################################
        elif self.experiments_name.startswith("synthetic"):
            dataset = SyntheticNormalDataset(seq_len=128, num=1000, D=self.D, random_seed=123)
            train_dataset, test_dataset = CPDDatasets.train_test_split_(
                dataset, test_size=0.3, shuffle=True
            )
            
        elif self.experiments_name == "topic_segmentation":
            dataset = TextSequenceDataset('/mnt/data/eromanenkova/Intern_CPD/Anna/text_segmentation/data_vecs_32.pt')
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

from torch.utils.data import Dataset, Subset

class TextSequenceDataset(Dataset):
    """Class for Dataset consists of sequences of MNIST images."""

    def __init__(self, path) -> None:
        """Initialize datasets' parameters.

        :param path_to_data: path to folders with MNIST sequences
        :param type_seq: type of data for loading (only normal, only anomaly, all)
        """
        super().__init__()

        # set paths to data
                
        self.data_vecs = torch.load(path)
        self.permutation = np.arange(len(self.data_vecs)//3) #np.random.permutation(np.arange(len(data_vecs)//3))    
        self.text = [self.data_vecs['text_'+str(i)] for i in self.permutation ]
        self.vecs = [self.data_vecs['t2v_'+str(i)] for i in self.permutation ]
        self.labels = [self.data_vecs['labels_'+str(i)] for i in self.permutation]
        
        
    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """Get one images' sequence and corresponding labels from dataset.

        :param idx: index of element in dataset
        :return: tuple of
             - sequence of images
             - sequence of labels
        """
        #seq_sentences = self.text[idx]
        seq_sentences = self.text[idx]
        seq_vecs = self.vecs[idx]
        seq_labels = self.labels[idx]
        return seq_vecs, seq_labels
    
    def get_text(self, idx):
        """Get one images' sequence and corresponding labels from dataset.

        :param idx: index of element in dataset
        :return: tuple of
             - sequence of images
             - sequence of labels
        """
        seq_sentences = self.text[idx]
        seq_vecs = self.vecs[idx]
        seq_labels = self.labels[idx]
        return seq_sentences, seq_vecs, seq_labels
    
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
        self.path_to_normal = os.path.join(path_to_data, "normal_data/")
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
                x1_ = abs(m.sample())
                x1.append(x1_)
            x1 = torch.stack(x1)

            m = MultivariateNormal(mu[1] * torch.ones(D), torch.eye(D))    
            x2 = []
            for _ in range(seq_len):
                x2_ = abs(m.sample())
                x2.append(x2_)
            x2 = torch.stack(x2)            
            
            #x1 = torch.normal(float(mu[0]), std, size=(seq_len, D))
            #x2 = torch.normal(float(mu[1]), std, size=(seq_len, D))
           
            x = torch.cat([x1[:idx], x2[idx:]])
            label = torch.cat([torch.zeros(idx), torch.ones(seq_len-idx)])
            data.append(x)
            labels.append(label)

        for idx in range(0, num - len(idxs_changes)):
            #mu = torch.randint(D, 100, (1, ))
            #x = torch.normal(float(mu), std, size=(seq_len, D))
            
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
    

#####################################################################################################
    
class BaselineDataset(Dataset):
    def __init__(self, cpd_dataset, baseline_type='simple', subseq_len=None):
        self.baseline_type = baseline_type
        
        def get_subset_(
            dataset: Dataset, subset_size: int, shuffle: bool = True, random_seed: int = 123
        ) -> Dataset:
            random.seed(random_seed)
            np.random.seed(random_seed)

            len_dataset = len(dataset)
            idx = np.arange(len_dataset)

            if shuffle:
                idx = random.sample(list(idx), min(subset_size, len_dataset))
            else:
                idx = idx[: subset_size]
            subset = Subset(dataset, idx)
            return subset  
    
    
        self.cpd_dataset = cpd_dataset
        self.N = len(cpd_dataset)
        # TODO FIX         
        self.T = len(cpd_dataset.__getitem__(0)[1])
        
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
            images = self.cpd_dataset[global_idx][0]
            images = images[local_idx]
            labels = self.cpd_dataset[global_idx][1][local_idx] 
            
        elif self.baseline_type == 'weak_labels':
            global_idx = idx // (self.T - self.subseq_len + 1)
            local_idx = idx %  (self.T - self.subseq_len + 1)
            images = self.cpd_dataset[global_idx][0][:, local_idx: local_idx + self.subseq_len]
            labels = max(self.cpd_dataset[global_idx][1][:, local_idx: local_idx + self.subseq_len])
            
        return images, labels    
