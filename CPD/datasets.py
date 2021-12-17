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

    def __init__(self, experiments_name: str) -> None:
        """Initialize class.

        :param experiments_name: type of experiments (only mnist available now!)
        """
        super().__init__()

        # TODO make 
        if experiments_name in [
            "human_activity",
            "mnist64", "mnist128", "mnist256", "mnist512", "mnist", 
            "1changes", "2changes", "4changes", "6changes", "8changes", "10changes",
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
        if self.experiments_name[:5] == "mnist":
            if self.experiments_name[5:] == '':
                self.experiments_name += '64'
            path_to_data = "/home/eromanenkova/Intern_CPD/Anna/" + "data/"+ self.experiments_name[5:] + "/"
            #path_to_data = "/home/eromanenkova/Intern_CPD/Anna/" + "data/64/"

            dataset = MNISTSequenceDataset(path_to_data=path_to_data, type_seq="all")
            train_dataset, test_dataset = CPDDatasets.train_test_split_(
                dataset, test_size=0.3, shuffle=True
            )
        if self.experiments_name[-7:] == "changes":
            path_to_data = "/home/eromanenkova/Intern_CPD/Anna/" + "data/"+ self.experiments_name + "/"
            #path_to_data = "/home/eromanenkova/Intern_CPD/Anna/" + "data/64/"

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
            
            train_dataset = HumanActivityDataset(path_to_data=path_to_data, type_="train")
            test_dataset = HumanActivityDataset(path_to_data=path_to_data, type_="test")
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
    

#TODO Reformat
class HumanActivityDataset(Dataset):

    def __init__(self, path_to_data, type_="train"):

        super().__init__()

        data, self.names = self.load_data(path_to_data, type_=type_)
        self.features, self.labels = self.preprocess(data, self.names)

        
    def load_data(self, path, type_="train"):
        data_path = path + "/T" + type_[1:] + "/X_" + type_ + ".txt"
        labels_path = path + "/T" + type_[1:] + "/y_" + type_ + ".txt"
        subjects_path = path + "/T" + type_[1:] + "/subject_id_" + type_ + ".txt"

        data = pd.read_csv(data_path, sep=" ", header=None)
        labels = pd.read_csv(labels_path, sep=" ", header=None)
        subjects = pd.read_csv(subjects_path, sep=" ", header=None)

        data["subject"] = subjects
        data["labels"] = labels
        
        df_names = pd.read_csv(path + '/features.txt', header=None)
        names = df_names[0]
        
        return data, names
    
    def preprocess(self, data, idxs):
    
        def find_change(array):
            count_ = 0
            changes = []
            len_ = len(array)
            for i in range(len_):             
                if i < len_-3:
                    value_current = array[i]
                    value_ahead = array[i+1]

                    if value_current!=value_ahead:
                        changes.append(i)
            return changes


        def find_idx(data):
            subjects = np.unique(data.subject)
            changes_global = np.array([])
            for subject in subjects:
                local = data[data.subject==subject]
                local = local.reset_index(drop=True)

                changes_ = find_change(local.labels)
                changes_ = np.append([0], changes_)
                start = 0
                len_ = len(changes_)
                changes_local = np.array([])
                while True:
                    changes_local = np.append(changes_local, changes_[start:start+4])
                    start += 2
                    if start > len_-4:
                        break
                changes_global = np.append(changes_global, changes_local)       
            return changes_global


        def prepare_idx(changes_global):
            max_len = 20

            changes_global = changes_global.reshape(changes_global.shape[0]//4, 4)
            changes_global[:, 0] += 1
            changes_global[0, 0] -= 1

            changes_global_final = []   

            for i, change in enumerate(changes_global):
                len_ = change[3] - change[0]
                diff = len_ - max_len
                if diff > 0:
                    if ((changes_global[i][1]-changes_global[i][0])<max_len) and (changes_global[i][1]-changes_global[i][0])>0:
                        changes_global_final.append([changes_global[i][0], changes_global[i][1], changes_global[i][2], changes_global[i][0] + max_len])
                    #else:
                    #    changes_global_final.append([changes_global[i][0], changes_global[i][0] + max_len, changes_global[i][0] + max_len, changes_global[i][0] + max_len])

            return changes_global_final


        def prepare_data(data, changes_global_final, idxs):
            max_len = int(changes_global_final[0][3] - changes_global_final[0][0])
            train_set = torch.tensor([])

            for i, change in enumerate(changes_global_final):
                start, end = int(change[0]), int(change[3])
                element = data[start:end].to_numpy()[:, :len(idxs)]
                if len(train_set) == 0:
                    train_set = torch.tensor(element)
                else:
                    train_set = torch.cat([train_set, torch.tensor(element)])

            train_set = train_set.reshape(len(train_set)//max_len, max_len, len(idxs))
            return train_set


        def prepare_labels(changes_global_final):
            labels = []
            max_len = int(changes_global_final[0][3] - changes_global_final[0][0])

            for change in changes_global_final:
                position = int(change[2]-change[0])
                if position < max_len:
                    labels_local = np.append(np.zeros(position), np.ones(max_len-position))
                    labels.append(labels_local)
            return torch.tensor(labels, dtype=torch.int)

        changes_global = find_idx(data)
        print(changes_global)
        changes_global_final = prepare_idx(changes_global)
        set_ = prepare_data(data, changes_global_final, idxs)
        labels = prepare_labels(changes_global_final)
        return set_, labels    
    
    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        print(idx)
        print('----')
        return self.features[idx], self.labels[idx]    
