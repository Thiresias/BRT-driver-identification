import os
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from collections import defaultdict
import cv2
from scipy.signal import medfilt
import scipy.signal as signal


class Load_au_fair(Dataset):
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
            if filename.endswith('AU.json'):
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
    

class Load_au_naive(Dataset):
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
        self.videos = defaultdict(lambda: [int, list()])
        self.total_duration = 0
        # Iterate through the directory and group frames by video
        for filename in os.listdir(self.directory):
            if filename.endswith('AU.json'):
                video_id = filename.split('_')[0]
                clip_name = filename.split('_')[:-1]
                clip_name = '_'.join(clip_name)  # Re-join video ID parts if it was split
                with open(os.path.join(self.directory, filename)) as f:
                    mat = json.load(f)
                    self.total_duration += len(mat)
                    if len(mat) >= seq_length:
                        self.videos[clip_name][0] = len(mat)
                        # print(self.directory)
                        # print(filename)
                        # print(self.videos[clip_name][1])
                        self.videos[clip_name][1].append(os.path.join(self.directory, filename))
                    else:
                        print(f"Clip {clip_name} is less than {seq_length//5} sec long.")

        self.video_ids = list(self.videos)

    def __len__(self):
        return len(self.video_ids)
    
    def get_list(self):
        return list(self.videos)


class Belkacem_CLS_AU(Dataset):
    def __init__(self, directory, seq_length=50, train=True):
        """
        Args:
            directory (string): Directory with all the .json files of the positive and negative class (pos=belkacem)
        """
        self.directory = directory
        self.seq_length = seq_length
        # # # Naive mode
        # self.videos = defaultdict(list)
        # # Fair mode
        self.videos = defaultdict(lambda: [int,defaultdict(list)])
        
        self.ds = Load_au_fair(self.directory, seq_length=self.seq_length, train=train)
        self.positive_samples = defaultdict(lambda: [int,defaultdict(list)])
        self.negative_samples = defaultdict(lambda: [int,defaultdict(list)])
        for vid in self.ds.get_list():
            self.videos[vid] = self.ds.videos[vid]
            if (('FF' in vid) or ('id' in vid)):
                self.negative_samples[vid] = self.ds.videos[vid]
            else:
                self.positive_samples[vid] = self.ds.videos[vid]
        self.video_ids = list(self.videos)
        self.total_duration = self.ds.total_duration
        


    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        # Fair mode
        dict_frames_paths = self.videos[video_id][1]
        list_of_clip = list(dict_frames_paths)
        random_clip_id = np.random.randint(0,len(list_of_clip))
        frames_paths = dict_frames_paths[list_of_clip[random_clip_id]][0]
        
        # Because of class imbalanced, select randomly a subset of negative samples

        vid_duration = self.videos[video_id][0]
        duration = 0
        frames = []
        # Load ldm file of the video
        # print(self.videos[video_id])
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
        label = 0.0 if (('FF' in video_id) or ('id' in video_id)) else 1.0
        # # Center # and reduce each features?
        # frames_tensor = frames_tensor - torch.mean(frames_tensor, dim=0)
        # frames_tensor = torch.norm(frames_tensor - torch.mean(frames_tensor, dim=1)[:, None, :], dim=2)
        # diff_frames_tensor = frames_tensor[:-1, :] - frames_tensor[1:, :]
        return frames_tensor, np.int64(label) , video_id, weight
    



class Belkacem_CLS_AU_balanced(Dataset):
    def __init__(self, directory, seq_length=50, train=True):
        """
        Args:
            directory (string): Directory with all the .json files of the positive and negative class (pos=belkacem)
        """
        self.directory = directory
        self.seq_length = seq_length
        # # # Naive mode
        # self.videos = defaultdict(list)
        # # Fair mode
        self.videos = defaultdict(lambda: [int,defaultdict(list)])
        
        self.ds = Load_au_fair(self.directory, seq_length=self.seq_length, train=train)
        self.positive_samples = defaultdict(lambda: [int,defaultdict(list)])
        self.negative_samples = defaultdict(lambda: [int,defaultdict(list)])
        for vid in self.ds.get_list():
            self.videos[vid] = self.ds.videos[vid]
            if ((len(vid) in [1,2]) or ('roop' in vid) or ('FF' in vid) or ('id' in vid) or ('cdf' in vid)):
                self.negative_samples[vid] = self.ds.videos[vid]
            else:
                self.positive_samples[vid] = self.ds.videos[vid]
        self.video_ids = list(self.positive_samples)
        self.total_duration = self.ds.total_duration
        if not train:
            print(self.negative_samples)


    def __len__(self):
        return len(list(self.positive_samples))

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        ## POSITIVE
        # Fair mode
        dict_frames_paths_pos = self.videos[video_id][1]
        list_of_clip_pos = list(dict_frames_paths_pos)
        random_clip_id_pos = np.random.randint(0,len(list_of_clip_pos))
        frames_paths_pos = dict_frames_paths_pos[list_of_clip_pos[random_clip_id_pos]][0]
        vid_duration_pos = self.videos[video_id][0]

        ## NEGATIVE
        # Because of class imbalanced, select randomly one negative sample
        list_of_neg = list(self.negative_samples)
        random_clip_id_neg = np.random.randint(0,len(self.negative_samples))
        frames_paths_neg = self.negative_samples[list_of_neg[random_clip_id_neg]][1][list(self.negative_samples[list_of_neg[random_clip_id_neg]][1])[0]][0]
        vid_duration_neg = self.negative_samples[list_of_neg[random_clip_id_neg]][0]
        
        ## ANCHOR
        list_of_anc = list(self.positive_samples)
        is_same=True
        while is_same:
            random_clip_id_anc = np.random.randint(0, len(self.positive_samples))
            if random_clip_id_anc != idx:
                is_same = False
        
        frames_paths_anc = self.positive_samples[list_of_anc[random_clip_id_anc]][1][list(self.positive_samples[list_of_anc[random_clip_id_anc]][1])[0]][0]
        vid_duration_anc = self.positive_samples[list_of_anc[random_clip_id_anc]][0]
        
        [pos_frames, pos_weight], [neg_frames, neg_weight], [anc_frames, anc_weight] = self.load_AU(frames_paths_pos, vid_duration_pos), self.load_AU(frames_paths_neg, vid_duration_neg), self.load_AU(frames_paths_anc, vid_duration_anc)


        # Stack frames into a single tensor
        frames_tensor_pos, frames_tensor_neg, frames_tensor_anc = torch.stack(pos_frames), torch.stack(neg_frames), torch.stack(anc_frames)
        frames_tensor = torch.stack([frames_tensor_pos, frames_tensor_neg])
        # diff_frames_tensor = frames_tensor[:-1, :, :] - frames_tensor[1:, :, :]
        # diff_frames_tensor = torch.norm(diff_frames_tensor, dim=2, p=2)
        # Get label
        label = [np.int64(1), np.int64(0)]
        # # Center # and reduce each features?
        # frames_tensor = frames_tensor - torch.mean(frames_tensor, dim=0)
        # frames_tensor = torch.norm(frames_tensor - torch.mean(frames_tensor, dim=1)[:, None, :], dim=2)
        frames_tensor = frames_tensor[:, :-1, :] - frames_tensor[:, 1:, :]

        video_id_pos = video_id
        video_id_neg = list_of_neg[random_clip_id_neg]
        return frames_tensor, torch.tensor(label) , [video_id_pos, video_id_neg], torch.tensor([pos_weight, neg_weight])


    def load_AU(self, frames_paths, vid_duration):
        frames = []
        # Load ldm file of the video
        # print(self.videos[video_id])
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
        return frames, weight