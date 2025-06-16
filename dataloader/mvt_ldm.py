import os
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from collections import defaultdict
import cv2
from scipy.signal import medfilt
import scipy.signal as signal


class Load_ldm_fair(Dataset):
    def __init__(self, directory, seq_length=40, train=True):
        """
        Args:
            directory (string): Directory with all the .json files.
        """
        if train:
            self.directory = os.path.join(directory, 'train')
        else:
            self.directory = os.path.join(directory, 'test')
        self.seq_length = seq_length
        self.videos = defaultdict(lambda: [int, defaultdict(list)])
        self.total_duration = 0
        # Iterate through the directory and group frames by video
        for filename in os.listdir(self.directory):
            if filename.endswith('ldm.json'):
                video_id = filename.split('_')[0]
                clip_name = filename.split('_')[:-1]
                clip_name = '_'.join(clip_name)  # Re-join video ID parts if it was split
                with open(os.path.join(self.directory, filename)) as f:
                    mat = json.load(f)
                    self.total_duration += len(mat)
                    if len(mat) >= seq_length:
                        if video_id in list(self.videos):
                            self.videos[video_id][0] = self.videos[video_id][0] + len(mat)
                            self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                        else:
                            self.videos[video_id][0] = len(mat)
                            self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                    else:
                        print(f"Clip {clip_name} is less than {seq_length//5} sec long.")

        self.video_ids = list(self.videos)

    def __len__(self):
        return len(self.video_ids)
    
    def get_list(self):
        return list(self.videos)
    

class Load_ldm_naive(Dataset):
    def __init__(self, directory, seq_length=40, train=True):
        """
        Args:
            directory (string): Directory with all the .json files.
        """
        if train:
            self.directory = os.path.join(directory, 'train')
        else:
            self.directory = os.path.join(directory, 'test')
        self.seq_length = seq_length
        self.videos = defaultdict(list)
        self.total_duration = 0
        # Iterate through the directory and group frames by video
        for filename in os.listdir(self.directory):
            if filename.endswith('ldm.json'):
                video_id = filename.split('_')[0]
                clip_name = filename.split('_')[:-1]
                clip_name = '_'.join(clip_name)  # Re-join video ID parts if it was split
                with open(os.path.join(self.directory, filename)) as f:
                    mat = json.load(f)
                    self.total_duration += len(mat)
                    if len(mat) >= seq_length:
                        self.videos[clip_name].append(os.path.join(self.directory, filename))
                    else:
                        print(f"Clip {clip_name} is less than {seq_length//5} sec long.")

        self.video_ids = list(self.videos)

    def __len__(self):
        return len(self.video_ids)
    
    def get_list(self):
        return list(self.videos)


class Belkacem_CLS(Dataset):
    def __init__(self, directory_pos, directory_neg, seq_length=50, train=True):
        """
        Args:
            directory_pos (string): Directory with all the .json files of the positive class (belkacem)
            directory_neg (string): Directory with all the .json files of the negative class (belkacem)
        """
        self.directory_pos = directory_pos
        self.directory_neg = directory_neg
        self.seq_length = seq_length
        # # # Naive mode
        # self.videos = defaultdict(list)
        # Fair mode
        self.videos = defaultdict(lambda: [int,defaultdict(list)])
        
        self.pos_ds = Load_ldm_fair(self.directory_pos, seq_length=self.seq_length, train=train)
        self.neg_ds = Load_ldm_fair(self.directory_neg, seq_length=self.seq_length, train=train)
        
        for vid in self.pos_ds.get_list():
            self.videos[vid] = self.pos_ds.videos[vid]
        for vid in self.neg_ds.get_list():
            self.videos[vid] = self.neg_ds.videos[vid]
        print(f"Combinaison: {list(self.videos)}")
        self.video_ids = list(self.videos)
        self.total_duration = self.pos_ds.total_duration + self.neg_ds.total_duration

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        # Fair mode
        vid_duration = self.videos[video_id][0]
        dict_frames_paths = self.videos[video_id][1]
        list_of_clip = list(dict_frames_paths)
        random_clip_id = np.random.randint(0,len(list_of_clip))


        frames_paths = dict_frames_paths[list_of_clip[random_clip_id]][0]
        # ## Naive mode
        # frames_paths = self.videos[video_id][0]
        frames = []
        # Load ldm file of the video
        with open(frames_paths) as f:
            mat = json.load(f)
            # print(len(mat))
            weight = vid_duration/ self.total_duration
            # Choose a random starting frame between first_frame and last_frame-length
            potential_starting_frame = range(len(mat) - (self.seq_length - 1))
            random_index = np.random.randint(0, len(potential_starting_frame))
            for index in range(random_index, random_index+self.seq_length):
                frame_ldm = torch.FloatTensor(mat[list(mat)[index]])
                frames.append(frame_ldm)


        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames)
        # diff_frames_tensor = frames_tensor[:-1, :, :] - frames_tensor[1:, :, :]
        # diff_frames_tensor = torch.norm(diff_frames_tensor, dim=2, p=2)
        # Get label
        label = 1.0 if 'belkacem' in frames_paths else 0.0
        # # Center # and reduce each features?
        frames_tensor = frames_tensor - torch.mean(frames_tensor, dim=1)[:, None, :]
        # frames_tensor = torch.norm(frames_tensor - torch.mean(frames_tensor, dim=1)[:, None, :], dim=2)
        diff_frames_tensor = frames_tensor[:-1, :] - frames_tensor[1:, :]
        return diff_frames_tensor, np.int64(label) , video_id, weight