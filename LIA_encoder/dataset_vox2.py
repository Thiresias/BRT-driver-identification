import os
import random
import glob
import cv2
import torch
import numpy as np
import pandas as pd


from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True




class Load_vid_fair(Dataset):
    def __init__(self, directory, seq_length=40, train=True, pos=True, obama=False):
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
            if filename.endswith('.mp4'):
                # video_id = "".join(filename.split('_')[0]) if not (('FF' in filename) or ('id' in filename))else f'{filename}'
                video_id = filename.split('__')[0]
                clip_name = filename
                cap = cv2.VideoCapture(os.path.join(self.directory, filename))
                nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = nb_frame/fps
                if fps >20:
                    if duration >= seq_length:
                        self.total_duration += duration
                        if video_id in list(self.videos):
                            self.videos[video_id][0] = self.videos[video_id][0] + duration
                            self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                        else:
                            self.videos[video_id][0] = duration
                            self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                    
                    else:
                        print(f"Clip {clip_name} duration is {duration} sec (<{seq_length} sec).")
                else:
                    print(f'Not enough fps (found {fps} need at least 20)')

                # if (fps == 30) or (fps == 25) or (np.abs(fps - 30)<0.1 or np.abs(fps-25)<0.1):
                #     if duration >= seq_length:
                #         self.total_duration += duration
                #         if video_id in list(self.videos):
                #             self.videos[video_id][0] = self.videos[video_id][0] + duration
                #             self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                #         else:
                #             self.videos[video_id][0] = duration
                #             self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                    
                #     else:
                #         print(f"Clip {clip_name} is less than {seq_length} sec long.")
                # else:
                #     print(f"Clip {clip_name} has a {fps} fps. Need to be 25 or 30.")

        self.video_ids = list(self.videos)

    def __len__(self):
        return len(self.video_ids)
    
    def get_list(self):
        return list(self.videos)
    


class Vox2dataset(Dataset):
    def __init__(self, directory, transform=None, seq_length=50, train=True, obama=False):
        """
        Args:
            directory (string): Directory with all the .json files of the positive and negative class (pos=belkacem)
        """
        self.directory = directory
        self.seq_length = seq_length
        self.dt = 0.2
        self.transform = transform

        self.classes = np.array([ 12,  15,  16,  18,  19,  20,  21,  22,  24,  25,  26,  27,  28,
        29,  32,  33,  36,  39,  40,  42,  43,  44,  46,  47,  49,  50,
        51,  52,  53,  55,  56,  58,  59,  60,  62,  64,  66,  68,  69,
        70,  71,  73,  75,  76,  78,  80,  82,  83,  84,  85,  86,  87,
        88,  89,  90,  91,  92,  96,  97,  98,  99, 100, 103, 104, 105,
       111, 117, 127, 128, 129, 137, 144, 145, 146, 149, 151, 155, 159,
       161, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176,
       177, 179, 180, 181, 184, 185, 186, 187, 188, 190, 191, 192, 195,
       197, 198, 201, 202, 203, 206, 220, 221, 223])
        # # Fair mode
        self.videos = defaultdict(lambda: [int,defaultdict(list)])
        self.positive_samples = Load_vid_fair(self.directory, seq_length=self.seq_length, train=train, pos=True, obama=obama).videos
        for vidname in list(self.positive_samples):
            self.videos[vidname] = self.positive_samples[vidname]
        self.video_ids = list(self.videos)
        self.K = 10
        print(f"In {'train' if train else 'test'} --> {len(self.videos)}")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames, labels, video_ids, fps_list = [], [], [], []
        for rdm_idx in range(self.K):
            np.random.seed(rdm_idx)
            ## Load a sample of sequence from videos
            dict_frames_paths = self.videos[video_id][1]
            list_of_clip = list(dict_frames_paths)
            random_clip_id = np.random.randint(0,len(list_of_clip))
            frames_paths = dict_frames_paths[list_of_clip[random_clip_id]][0]
            vid_duration = self.videos[video_id][0]

            # [sequence, fps] = self.load_vid(frames_paths, vid_duration, self.transform) # BRT-DFD baseline
            [sequence, fps] = self.load_vid_multiscale(frames_paths, vid_duration, self.transform) # BRT-DFD+
            label = int(video_id[2:])

            frames.append(sequence)
            labels.append(label)
            video_ids.append(video_id)
            fps_list.append(fps)
        sorted_labels = np.unique(labels)
        newlabels = np.array([np.where(x == self.classes)[0][0] for x in labels])
        
        frames_tensor = torch.stack(frames)
        label_tensor = torch.tensor(newlabels)
        fps_tensor = torch.tensor(fps_list)


        return frames_tensor, label_tensor, video_ids, fps_tensor
    
    def load_vid(self, frames_paths, vid_duration, transform=None):
        frames = []
        # Load video
        cap = cv2.VideoCapture(frames_paths)
        nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sampling = 5 if np.abs(fps-25)<np.abs(fps-30) else 6

        # Choose a random starting frame between first_frame and last_frame-length
        potential_starting_frame = range(int(nb_frame - (self.seq_length*round(fps))))
        random_index = np.random.randint(0, len(potential_starting_frame)+1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_index-1)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count == (self.seq_length)*round(fps):
                break
            if (count % sampling == 0) and ret:
                frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(torch.tensor(frame_RGB))
            count += 1
        try:
            del frame_RGB
        except:
            print("No frame to delete")

        cap.release()
        vid = torch.stack(frames).permute(0, 3, 1, 2)
        return vid, 1
    
    def load_vid_multiscale(self, frames_paths, vid_duration, transform=None):
        frames = []
        MAX_FRAME = 40
        # Load video
        cap = cv2.VideoCapture(frames_paths)
        nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sampling = 5 if np.abs(fps-25)<np.abs(fps-30) else 6


        # Find the maximum frame interval
        max_intervall = (nb_frame-MAX_FRAME)//(MAX_FRAME-1)
        
        # Choose a random starting frame between first_frame and last_frame-length
        random_interval = np.random.randint(1,np.max([np.min([max_intervall-1, 5]), 2]))
        potential_starting_frame = range(int(nb_frame - (random_interval*(MAX_FRAME-1)+MAX_FRAME) ))
        random_index = np.random.randint(0, len(potential_starting_frame)+1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_index-1)
        count = 0
        accepted_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if accepted_frames == MAX_FRAME:
                break
            if (count % random_interval == 0) and ret:
                frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                accepted_frames += 1
                frames.append(torch.tensor(frame_RGB))
            count += 1
        try:
            del frame_RGB
        except:
            print("No frame to delete")

        cap.release()
        vid = torch.stack(frames).permute(0, 3, 1, 2)
        return vid, fps