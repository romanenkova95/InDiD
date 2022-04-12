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
from CPD import models
from torchvision import transforms
from collections import defaultdict
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm
import av
import pickle

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)



class CPDDatasets:
    """Class for experiments' datasets."""

    def __init__(self, experiments_name: str, 
                 model_type='seq2seq', random_seed=123) -> None:
        """Initialize class.

        :param experiments_name: type of experiments (only mnist available now!)
        """
        super().__init__()
        self.random_seed = random_seed
        # TODO make 
        if experiments_name in [
            "human_activity",
            "mnist",
            "explosion",
            "oops",
            "road_accidents"
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
                dataset, test_size=0.3, shuffle=True, random_seed=self.random_seed
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
                dataset, test_size=0.3, shuffle=True, random_seed=self.random_seed
            )
            
        elif self.experiments_name == "human_activity":
            path_to_data = "data/human_activity/"
            train_dataset = HumanActivityDataset(path_to_data=path_to_data, seq_len=20, train_flag=True)
            test_dataset = HumanActivityDataset(path_to_data=path_to_data, seq_len=20, train_flag=False)
            
        elif self.experiments_name == "explosion":
            path_to_data = "data/explosion/"
            # https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
            
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            side_size = 256
            crop_size = 256

            transform=Compose([
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ])
            

            train_dataset = UCFVideoDataset(clip_length_in_frames=16, step_between_clips=5, 
                                            path_to_data=path_to_data, 
                                            path_to_annotation='UCF_train_time_markup.txt', 
                                            video_transform=transform,
                                            num_workers=0, fps=30, sampler='equal')
            test_dataset = UCFVideoDataset(clip_length_in_frames=16, step_between_clips=16, 
                                           path_to_data=path_to_data, 
                                           path_to_annotation='UCF_test_time_markup.txt', 
                                           video_transform=transform,
                                           num_workers=0, fps=30, sampler='downsample_norm')
        elif self.experiments_name == "road_accidents":
            path_to_data = "data/road_accidents/"
            # https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            side_size = 256
            crop_size = 256

            transform=Compose([
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ])
            
            train_dataset = UCFVideoDataset(clip_length_in_frames=16, step_between_clips=5, 
                                            path_to_data=path_to_data, 
                                            path_to_annotation='UCF_road_train_time_markup.txt', 
                                            video_transform=transform,
                                            num_workers=0, fps=30, sampler='equal')
            test_dataset = UCFVideoDataset(clip_length_in_frames=16, step_between_clips=16, 
                                           path_to_data=path_to_data, 
                                           path_to_annotation='UCF_road_test_time_markup.txt', 
                                           video_transform=transform,
                                           num_workers=0, fps=30, sampler='downsample_norm')
            
        return train_dataset, test_dataset

    @staticmethod
    def train_test_split_(
        dataset: Dataset, test_size: float = 0.3, shuffle: bool = True, random_seed: int = 123
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset on train and test.

        :param dataset: dataset for splitting
        :param test_size: size of test data
        :param shuffle: if True, shuffle data
        :return: tuple of
            - train dataset
            - test dataset
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
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
        dataset: Dataset, subset_size: int, shuffle: bool = True, random_seed: int = 123
    ) -> Dataset:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
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

        # normalize images
        seq_images = seq_images / 255
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
        models.fix_seeds(random_seed)

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
    
class UCFVideoDataset(Dataset):
    def __init__(self,
             clip_length_in_frames,
             step_between_clips,
             path_to_data,
             path_to_annotation,
             video_transform=None, 
             num_workers=0, fps=30, sampler='all'):
        
        super().__init__()

        self.clip_length_in_frames = clip_length_in_frames
        self.step_between_clips = step_between_clips
        self.num_workers = num_workers
        self.fps = fps
        
        # IO
        self.path_to_data = path_to_data
        self.video_list = self._get_video_list(dataset_path=self.path_to_data)

        # annotation loading
        dict_metadata, dict_types = self._parce_annotation(path_to_annotation)    
        
        path_to_clips = '{}_clips_len_{}_step_{}_fps_{}.pth'.format(self.path_to_data.split('/')[1],
                                                                    self.clip_length_in_frames, 
                                                                    self.step_between_clips, 
                                                                    self.fps)
        if 'train' in path_to_annotation:
            path_to_clips = 'train_' + path_to_clips
        else:
            path_to_clips = 'test_' + path_to_clips            
            
        if os.path.exists(path_to_clips):
            with open(path_to_clips, 'rb') as clips:
                self.video_clips = pickle.load(clips)
        else:
            # data loading
            self.video_clips = VideoClips(video_paths=self.video_list,
                                          clip_length_in_frames=self.clip_length_in_frames,
                                          frames_between_clips=self.step_between_clips, 
                                          frame_rate = self.fps,
                                          num_workers=self.num_workers)
        
            # labelling
            self.video_clips.compute_clips(self.clip_length_in_frames, self.step_between_clips, self.fps)
            self._set_labels_to_clip(dict_metadata)
        
        # transforms and samplers
        self.video_transform=video_transform
        self.sampler = sampler
        
        # save dataset
        if not os.path.exists(path_to_clips):
            with open(path_to_clips, 'wb') as clips:
                pickle.dump(self.video_clips, clips, protocol=pickle.HIGHEST_PROTOCOL)                
        
        if sampler=='equal':
            self.video_clips.valid_idxs = self._equal_sampling()
        elif sampler=='downsample_norm':
            self.video_clips.valid_idxs = self._equal_sampling(downsample_normal=300)
        elif sampler=='all':
            pass
        else:
            raise ValueError('Wrong type of sampling')
            
    
    def __len__(self):
        return len(self.video_clips.valid_idxs)
        
    def __getitem__(self, idx):
        idx = self.video_clips.valid_idxs[idx]
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        label = np.array(self.video_clips.labels[video_idx][clip_idx], dtype=int)
        # shoud be channel, seq_len, height, width
        video = video.permute(3, 0, 1, 2)        
        if self.video_transform is not None:
            video = self.video_transform(video) 
        return video, label

    def _set_labels_to_clip(self, dict_metadata):
        #self.video_clips.labels = []
        self.video_clips.labels = []
        self.video_clips.valid_idxs = list(range(0, len(self.video_clips)))
        self.video_clips.normal_idxs = defaultdict(list)
        self.video_clips.cp_idxs = defaultdict(list)            
        
        global_clip_idx = -1
        for video_idx, vid_clips in tqdm(enumerate(self.video_clips.clips), total=len(self.video_clips.clips)):
            video_labels = []
            
            video_path = self.video_clips.video_paths[video_idx]
            video_name = os.path.basename(video_path)
            
            # get time unit to map frame with its time appearance
            time_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
            
            # get start, change point, end from annotation
            annotated_time = dict_metadata[video_name]            
            
            cp_video_idx = len(vid_clips)
                
            for clip_idx, clip in enumerate(vid_clips):
                clip_labels = []                
                global_clip_idx += 1
                
                clip_start = float(time_unit * clip[0].item())
                clip_end = float(time_unit * clip[-1].item())
                
                not_suit_flag = True   
                for start_time, change_point, end_time in annotated_time:
                    if end_time != -1.0:
                        if (clip_end > end_time) or (clip_start > end_time) or (clip_start < start_time):
                            continue
                    else:
                        if clip_start < start_time:
                            continue
                    if (clip_start > change_point) and (change_point != -1.0):
                        # "abnormal" clip appears after change point                            
                        continue
                    else:
                        not_suit_flag = False
                        # proper clip                        
                        break
                                           
                if not_suit_flag:
                    # drop clip idx from dataset 
                    video_labels.append([])
                    self.video_clips.valid_idxs.remove(global_clip_idx)
                else:
                    if 'Normal' in video_path:
                        clip_labels = list(np.zeros(len(clip)))
                        self.video_clips.normal_idxs[video_path].append(global_clip_idx)
                    else:
                        for frame in clip:
                            frame_time = float(time_unit * frame.item())
                            # due to different rounding while moving from frame to time
                            # the true change point is delayed by ~1 frame
                            # so, we've added the little margin
                            if (frame_time >= change_point - 1e-6) and (change_point != -1.0):
                                clip_labels.append(1)
                            else:
                                clip_labels.append(0)
                        if sum(clip_labels) > 0:
                            self.video_clips.cp_idxs[video_path].append(global_clip_idx)                            
                        else:
                            self.video_clips.normal_idxs[video_path].append(global_clip_idx)
                    video_labels.append(clip_labels)
            self.video_clips.labels.append(video_labels)
        return self 
        
    def _get_video_list(self, dataset_path):
        assert os.path.exists(dataset_path), "VideoIter:: failed to locate: `{}'".format(dataset_path)
        vid_list = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                if 'mp4' not in name:
                    continue
                vid_list.append(os.path.join(path, name))
        return vid_list             
    
    def _parce_annotation(self, path_to_annotation):
        dict_metadata = defaultdict(list)
        dict_types = defaultdict(list)

        with open(path_to_annotation) as f:
            metadata = f.readlines()
            for f in metadata:
                #parce annotation 
                f = f.replace('\n', '') 
                f = f.split('  ')
                video_name = f[0]
                video_type = f[1]        
                change_time = float(f[3])
                video_borders = (float(f[2]), float(f[4]))
                dict_metadata[video_name].append((video_borders[0], change_time, video_borders[1]))
                dict_types[video_name].append(video_type)
        return dict_metadata, dict_types 
    
    def _equal_sampling(self, downsample_normal=None):
        """Balance sets with and without changes so that the equal number 
        of clips are sampled from each video (if possible)."""
        
        def _get_indexes(dict_normal, dict_cp):
            """Get indexes of clips from totally normal video and from video with anomaly"""
            cp_idxs = list(np.concatenate([idx for idx in dict_cp.values()]))

            normal_idxs = []
            normal_cp_idxs = []

            for k,v in zip(dict_normal.keys(), dict_normal.values()):
                if k in dict_cp.keys():
                    normal_cp_idxs.extend(v)
                else:
                    normal_idxs.extend(v)
            return cp_idxs, normal_idxs, normal_cp_idxs
        
        def _uniform_sampling(paths, dict_for_sampling, max_samples):
            sample_idxs = []
            for path in paths:
                idxs = dict_for_sampling[path]                
                if (len(idxs) > max_samples):
                    step = len(idxs) // max_samples                    
                    sample_idxs.extend(idxs[::step][:max_samples])
                else:
                    sample_idxs.extend(idxs[:max_samples])   
            return sample_idxs

        def _random_sampling(idxs_for_sampling, max_samples, random_seed=123):
            np.random.seed(random_seed)
            random.seed(random_seed)
            sample_idxs = random.choices(idxs_for_sampling, k=max_samples)
            return sample_idxs


        sample_idxs = []

        cp_paths = self.video_clips.cp_idxs.keys()
        normal_paths = [x for x in self.video_clips.normal_idxs.keys() if x not in cp_paths]
        normal_cp_paths = [x for x in self.video_clips.normal_idxs.keys() if x in cp_paths]
                
        cp_idxs, normal_idxs, normal_cp_idxs = _get_indexes(self.video_clips.normal_idxs, self.video_clips.cp_idxs)
        cp_number = len(cp_idxs)
        
        
        if downsample_normal is not None:
            max_samples = downsample_normal
        else:
            max_samples = cp_number
        
        # sample ~50% of normal clips from video with change point
        if len(cp_idxs) > len(normal_cp_paths):
            normal_from_cp = _uniform_sampling(paths=normal_cp_paths, 
                                               dict_for_sampling=self.video_clips.normal_idxs, 
                                               max_samples=max_samples // (len(normal_cp_paths) * 2))
        else:
            print('Equal sampling is impossible, do random sampling.')
            normal_from_cp = _random_sampling(idxs_for_sampling=normal_cp_idxs, 
                                             max_samples=max_samples // 2)

        # sample ~50% of normal clips from normal video
        max_rest = (max_samples - len(normal_from_cp)) 
        
        if max_rest > len(self.video_clips.normal_idxs.keys()):
            normal = _uniform_sampling(paths=normal_paths, 
                                       dict_for_sampling=self.video_clips.normal_idxs, 
                                       max_samples=max_rest // len(normal_paths))

        else:
            print('Equal sampling is impossible, do random sampling.')
            normal = _random_sampling(idxs_for_sampling=normal_idxs, 
                                     max_samples=max_rest)
            
        # sometimes it's still not enough because of different video length
        if len(normal_from_cp) + len(normal) < max_samples:
            extra = _random_sampling(idxs_for_sampling=normal_idxs + normal_cp_idxs, 
                                     max_samples=max_samples-len(normal_from_cp)-len(normal))
        else:
            extra = []
        sample_idxs = cp_idxs + normal_from_cp + normal + extra
        return sample_idxs  
    
#######################################################################################################
# https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch/
def to_tensor(clip):
    return clip.float().permute(3, 0, 1, 2) / 255.0

def normalize(clip, mean, std, inplace=False):
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip    

def resize(clip, target_size, interpolation_mode):
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )    

class ToTensorVideo(object):
    def __init__(self):
        pass

    def __call__(self, clip):
        return to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__
    

class NormalizeVideo(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)
    
    
class ResizeVideo:
    def __init__(
            self,
            size,
            interpolation_mode="bilinear"
    ):
        self.size = size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        return resize(clip, self.size, self.interpolation_mode)        
    
    
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
            images = images[:, local_idx]
            labels = self.cpd_dataset[global_idx][1][local_idx] 
            
        elif self.baseline_type == 'weak_labels':
            global_idx = idx // (self.T - self.subseq_len + 1)
            local_idx = idx %  (self.T - self.subseq_len + 1)
            images = self.cpd_dataset[global_idx][0][:, local_idx: local_idx + self.subseq_len]
            labels = max(self.cpd_dataset[global_idx][1][:, local_idx: local_idx + self.subseq_len])
            
        return images, labels    


class TSCPDataset(Dataset):
    def __init__(self, experiments_name, window, train_flag=True):
        self.window = window
        self.cpd_dataset = CPDDatasets(experiments_name, model_type='seq2seq', random_seed=123).get_dataset_()
        
        if train_flag:
            self.cpd_dataset = self.cpd_dataset[0]
        else:
            self.cpd_dataset = self.cpd_dataset[1]            
        
    def __len__(self):
        return len(self.cpd_dataset)
        
    def __getitem__(self, idx):
        images, labels = self.cpd_dataset.__getitem__(idx)
        history = images[:self.window]
        future = images[self.window:2*self.window]
        # TODO think how to rewrite
        #labels = np.argmax(labels)
            
        return (history, future), labels    

