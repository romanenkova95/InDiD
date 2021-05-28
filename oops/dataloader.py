import json
import os
import numpy as np
import statistics
from argparse import Namespace
from glob import glob

import av
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

import py12transforms as T
from sampler import DistributedSampler, UniformClipSampler, RandomClipSampler

normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
unnormalize = T.Unnormalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
train_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    T.RandomHorizontalFlip(),
    normalize,
    T.RandomCrop((112, 112))
])
test_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    normalize,
    T.CenterCrop((112, 112))
])

from torch.utils.data import Sampler
from typing import Optional, List, Iterator, Sized, Union, cast
class RandomClipSampler_v2(Sampler):
    def __init__(self, video_clips: VideoClips, max_clips_per_video: int) -> None:
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self) -> Iterator[int]:
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs[:1000])
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs_))
        return iter(idxs_[perm].tolist())

    def __len__(self) -> int:
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)


class KineticsAndFails(VisionDataset):
    FLOW_FPS = 8

    def __init__(self, fails_path, frames_per_clip, step_between_clips, fps, transform=None,
                 video_clips=None, get_clip_times=False, fails_video_list=None,
                 flow_histogram=False, **kwargs):

        self.fps = fps
        self.t = transform
        self.flow_histogram = flow_histogram
        self.video_clips = None
        self.fails_path = fails_path
        self.get_clip_times = get_clip_times

        # load prepared video
        if video_clips:
            self.video_clips = video_clips
        else:
            assert fails_path is None or fails_video_list is None
            video_list = fails_video_list or glob(os.path.join(fails_path, '**', '*.mp4'), recursive=True)
            self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips, fps)

        # load metadata
        with open("transition_times.json") as f:
            self.fails_data = json.load(f)

        # load raw video
        if video_clips is None:
            idxs = []
            for i, video_path in enumerate(self.video_clips.video_paths):
                video_path = os.path.splitext(os.path.basename(video_path))[0]
                if video_path in self.fails_data:
                    idxs.append(i)
            self.video_clips = self.video_clips.subset(idxs)
            self.video_clips.compute_clips(frames_per_clip, step_between_clips, fps)

        self.video_clips.labels = []
        # labeled video
        for video_idx, vid_clips in tqdm(enumerate(self.video_clips.clips), total=len(self.video_clips.clips)):
            video_path = self.video_clips.video_paths[video_idx]
            t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
            # at least three time of changing, choose one
            t_fail = sorted(self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['t'])
            t_fail = t_fail[len(t_fail) // 2]

            # check that not -1
            if t_fail < 0 or not 0.01 <= statistics.median(
                    self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['rel_t']) <= 0.99 or \
                    self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['len'] < 3.2 or \
                    self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['len'] > 30:
                self.video_clips.clips[video_idx] = torch.Tensor()
                self.video_clips.resampling_idxs[video_idx] = torch.Tensor()
                self.video_clips.labels.append([])
                continue

            # clip = [frames], video = [clips], labeling by frame
            labels_per_clips = []
            for clip_idx, clip in enumerate(vid_clips):
                labels_per_frame = []
                for frame in clip:
                    t_start = float(t_unit * frame.item())
                    if t_start <= t_fail:
                        label = 0
                    elif t_start > t_fail:
                        label = 1
                    labels_per_frame.append(label)
                labels_per_clips.append(labels_per_frame)
            self.video_clips.labels.append(labels_per_clips)

        clip_lengths = torch.as_tensor([len(v) for v in self.video_clips.clips])
        self.video_clips.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.video_clips.num_clips()

    def compute_clip_times(self, video_idx, clip_idx):
        video_path = self.video_clips.video_paths[video_idx]
        video_path = os.path.join(self.fails_path, os.path.sep.join(video_path.rsplit(os.path.sep, 2)[-2:]))
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
        t_start = float(t_unit * start_pts)
        t_end = float(t_unit * end_pts)
        return t_start, t_end

    def __getitem__(self, idx):
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        video_path = self.video_clips.video_paths[video_idx]
        try:
            label = np.transpose(self.video_clips.labels[video_idx][clip_idx])
        except:
            label = -1

        if self.t is not None:
            video = self.t(video)

        t_start = t_end = -1
        if self.get_clip_times:
            t_start, t_end = self.compute_clip_times(video_idx, clip_idx)

        return video, label, (video_path, t_start, t_end)


def get_video_loader(**kwargs):
    args = Namespace(**kwargs)
    args.fails_video_list = None
    args.cache_dataset = True
    #if args.val:
    #    args.fails_path = os.path.join(args.fails_path, 'val')
    #else:
    args.fails_path = os.path.join(args.fails_path, 'train')

    clips = None

    cache_path = os.path.join(args.dataset_path,
                              '{0}_videoclips.pth'.format('val' if args.val else 'train'))
    if args.cache_dataset and os.path.exists(cache_path):
        clips = torch.load(cache_path)
        if args.local_rank <= 0:
            print(f'Loaded dataset from {cache_path}')
    fns_to_remove = None
    
    dataset = KineticsAndFails(video_clips=clips, fns_to_remove=fns_to_remove, **vars(args))
    dataset = remove_unnecessary_clips(dataset)
    
        
       
    args.sample_all_clips = False

    if not args.val:
        print(f'Dataset contains {len(dataset)} items')
    if args.cache_dataset and args.local_rank <= 0 and clips is None:
        torch.save(dataset.video_clips, cache_path)

    if args.val:
        sampler = UniformClipSampler(dataset.video_clips, 2500)
    else:
        sampler = RandomClipSampler(dataset.video_clips, 2500 if args.sample_all_clips else args.clips_per_video)

    
    if args.local_rank != -1:
        sampler = DistributedSampler(sampler)
    return data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        drop_last=False
    )

def remove_unnecessary_clips(dataset):
    clips = dataset.video_clips

    new_labels = []
    new_clips = []
    new_paths = []
    new_pts = []
    new_fps = []
    new_resampling_idx = []

    good = 0
    without_labels = 0
    only_ones = 0
    indices = []
    for i in range(len(clips.clips)):
        curr_clips = []
        curr_labels = []
        curr_resampling_idx = []
        if len(clips.labels[i]) != 0:
            for j in range(len(clips.clips[i])):
                    sum_l = np.sum(clips.labels[i][j])
                    if (sum_l == 16):
                        only_ones += 1
                    else:
                        good += 1
                        curr_clips.append(clips.clips[i][j])
                        curr_labels.append(clips.labels[i][j])
                        curr_resampling_idx.append(clips.resampling_idxs[i][j])
            indices.append(i)
            new_labels.append(curr_labels)
            new_clips.append(curr_clips)
            new_paths.append(clips.video_paths[i])
            new_pts.append(clips.metadata['video_pts'][i])
            new_fps.append(clips.metadata['video_fps'][i])
            new_resampling_idx.append(curr_resampling_idx)      
        else:
            without_labels += 1

    print('Good clips:', good)        
    print('Only ones:', only_ones)    
    print('Without labels:', without_labels)

    dataset.video_clips.clips = new_clips
    dataset.video_clips.labels = new_labels
    dataset.video_clips.video_paths = new_paths
    dataset.video_clips.metadata['video_paths'] = new_paths
    dataset.video_clips.metadata['video_pts'] = new_pts
    dataset.video_clips.metadata['video_fps'] = new_fps
    clip_lengths = torch.as_tensor([len(v) for v in dataset.video_clips.clips])    
    dataset.video_clips.cumulative_sizes = clip_lengths.cumsum(0).tolist()
    dataset.video_clips.resampling_idxs = new_resampling_idx
    return dataset

