import os
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from collections import defaultdict
import cv2
from scipy.signal import medfilt
import scipy.signal as signal


class VideoFramesDataset(Dataset):
    def __init__(self, directory):
        """
        Args:
            directory (string): Directory with all the .mat files.
        """
        self.directory = directory
        self.videos = defaultdict(list)

        # Iterate through the directory and group frames by video
        for filename in os.listdir(directory):
            if filename.endswith('head-pose.json'):
                video_id= filename.split('_')[:-1]
                video_id = '_'.join(video_id)  # Re-join video ID parts if it was split
                self.videos[video_id].append(os.path.join(directory, filename))

        self.video_ids = list(self.videos)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames_paths = self.videos[video_id][0]
        frames = []

        # Load each frame for the video
        with open(frames_paths) as f:
            mat = json.load(f)["frame_1"]
            for frame in range(len(mat['yaw'])):
                frame_yaw = mat['yaw'][frame].flatten()
                # print(frame_pos.shape)
                frame_pitch = mat['pitch'][frame].flatten()
                # print(frame_trans.shape)
                frame_roll = mat['roll'][frame].flatten()
                frame_data = np.concatenate((frame_yaw, frame_pitch, frame_roll), axis=0)
                frames.append(torch.tensor(frame_data, dtype=torch.float32))
                # frames.append(torch.tensor(frame_pos, dtype=torch.float32))

        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames)

        # Get label
        id = frames_paths.split('_')[0]
        num = int(id[2:])
        label = num

        # Center and reduce each features
        frames_tensor = frames_tensor - torch.mean(frames_tensor, dim=0)/torch.std(frames_tensor, dim=0)
        
        return frames_tensor, np.float32(label), video_id





class MvtAnalysis(Dataset):
    def __init__(self, directory_AU, directory_HP, train=False):
        """
        Args:
            directory (string): Directory with all the .mat files.
        """
        if train:
            self.directory_AU = os.path.join(directory_AU, 'train')
            self.directory_HP = os.path.join(directory_HP, 'train')
        else:
            self.directory_AU = os.path.join(directory_AU, 'test')
            self.directory_HP = os.path.join(directory_HP, 'test')
        self.videos_AU = defaultdict(list)
        self.videos_HP = defaultdict(list)

        # Iterate through the directory and group frames by video
        for filename_AU in os.listdir(self.directory_AU):
            equivalent_HP = f"{filename_AU[:-7]}correctedHP.json"
            path_HP = os.path.join(self.directory_HP, equivalent_HP)
            if filename_AU.endswith('AU.json') and os.path.exists(path_HP):
                with open(os.path.join(self.directory_AU, filename_AU), 'r') as f:
                    json_file_AU = json.load(f)
                with open(path_HP,'r') as f:
                    json_file_HP = json.load(f)
                    if len(list(json_file_AU)) == 40 and len(list(json_file_HP)) == 40:
                        video_id = filename_AU.split('_')[:-1]
                        video_id = '_'.join(video_id)  # Re-join video ID parts if it was split
                        self.videos_AU[video_id].append(os.path.join(self.directory_AU, filename_AU))

                        filename_HP = os.path.basename(path_HP)
                        video_id = filename_HP.split('_')[:-1]
                        video_id = '_'.join(video_id)  # Re-join video ID parts if it was split
                        self.videos_HP[video_id].append(os.path.join(self.directory_HP, filename_HP))
                    else:
                        print(f"Video {filename_AU} or {filename_HP} doesn't contain enough frames for {'train' if train else 'test'} set.")
            # if filename_HP.endswith('correctedHP.json'):
            #     with open(os.path.join(self.directory_HP, filename_HP), 'r') as f:
            #         json_file = json.load(f)
            #         if len(list(json_file)) == 40:
            #             video_id = filename_HP.split('_')[:-1]
            #             video_id = '_'.join(video_id)  # Re-join video ID parts if it was split
            #             self.videos_HP[video_id].append(os.path.join(self.directory_HP, filename_HP))
            #         else:
            #             print(f"Video {filename_AU} doesn't contain enough frames for {'train' if train else 'test'} set.")

        self.video_ids = list(self.videos_AU)
        print(self.video_ids)
    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames_paths_AU = self.videos_AU[video_id][0]
        au_list = []
        
        # Load each frame for the video
        with open(frames_paths_AU, 'r') as f:
            au_dict = json.load(f)
        for frame in list(au_dict):
            au = au_dict[frame]
            au_list.append(torch.tensor(au[3:7]))
            

        # Stack frames into a single tensor
        frames_tensor_AU = torch.stack(au_list)

        # Get label
        vid_name = os.path.basename(frames_paths_AU) # path/to/idX_vidY_AU.json
        id = vid_name.split('_')[0]     # idX_vidY_AU.json
        num = int(id[2:])               # idX
        label = CDFlabel_transfo(num)   # X'
        label = int(label)

        # Apply median filter across the temporal dimension (T, n)
        kernel_size = 7
        frames_tensor_AU_np = frames_tensor_AU.numpy()  # Convert to numpy for processing
        frames_tensor_AU_np = medfilt(frames_tensor_AU_np, kernel_size=[kernel_size, 1])
        frames_tensor_AU = torch.tensor(frames_tensor_AU_np)  # Convert back to torch tensor

        # # Design a high-pass Butterworth filter
        # sample_rate, cutoff_frequency = 5, 1/4 # 5Hz
        # nyquist = 0.5 * sample_rate
        # normal_cutoff = cutoff_frequency / nyquist
        # b, a = signal.butter(N=4, Wn=normal_cutoff, btype='high', analog=False)

        # frames_paths_HP = self.videos_HP[video_id][0]
        # hp_list = []
        # with open(frames_paths_HP, 'r') as f:
        #     hp_dict = json.load(f)
        # for frame in list(hp_dict):
        #     yaw, pitch, roll = hp_dict[frame]['yaw'], hp_dict[frame]['pitch'], hp_dict[frame]['roll']
        #     hp_list.append(torch.tensor([yaw, pitch, roll]))
        # frames_tensor_HP = torch.stack(hp_list)
        # # Initialize an output matrix with the same shape
        # filtered_matrix = np.zeros_like(frames_tensor_AU)

        
        # # Apply the filter to each channel
        # for i in range(frames_tensor_AU.shape[1]):  # Iterate over the second dimension (features/channels)
        #     filtered_matrix[:, i] = signal.filtfilt(b, a, frames_tensor_AU[:, i], padlen=max(len(frames_tensor_AU[:, i])-1, 0))
        # frames_tensor_AU = torch.tensor(filtered_matrix, dtype=torch.float32)

        # # Center HP features
        # frames_tensor_HP = frames_tensor_HP - torch.mean(frames_tensor_HP, dim=0)

        # Concatenate AU features and HP features together
        # frames_tensor = torch.cat((frames_tensor_AU, frames_tensor_HP), dim=1)
        
        
        
        return frames_tensor_AU, label, video_id
    
    def _get_nb_POI(self):
        vid_id = [filename.split('_')[0] for filename in self.video_ids]
        return len(np.unique(vid_id))
    
def filter_short_videos(orig_dict):
    new_dict = defaultdict()
    vids = list(orig_dict)
    for vid in vids:
        print(orig_dict[vid])
        if len(orig_dict[vid]) != 40:
            print(f"Video {vid} doesn't contain enough frames.")
        else:
            new_dict[vid] = orig_dict[vid]
    return new_dict

def CDFlabel_transfo(id):
    if id <= 11:
        return id
    elif id <= 13:
        return id - 1
    elif id <= 17:
        return id - 3
    elif id <= 41:
        return id - 4
    elif id < 43:
        return id - 5
    else:
        return id - 6