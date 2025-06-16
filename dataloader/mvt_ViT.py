import os
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from collections import defaultdict
import cv2
from scipy.signal import medfilt
import scipy.signal as signal
from facenet_pytorch import MTCNN
from PIL import Image

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)


class Load_vid_fair(Dataset):
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
            if filename.endswith('.mp4'):
                video_id = filename.split('_')[0] if '_' in filename else f'FFtrain{filename[:-4]}'
                clip_name = filename.split('_')[:-1] if '_' in filename else f'FFtrain{filename[:-4]}_part_1'
                clip_name = '_'.join(clip_name)  if '_' in filename else clip_name# Re-join video ID parts if it was split
                cap = cv2.VideoCapture(os.path.join(self.directory, filename))
                nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = nb_frame/fps
                if duration > seq_length:
                    self.total_duration += duration
                    if video_id in list(self.videos):
                        self.videos[video_id][0] = self.videos[video_id][0] + duration
                        self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                    else:
                        self.videos[video_id][0] = duration
                        self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                else:
                    print(f"Clip {clip_name} is less than {seq_length} sec long.")

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


class Belkacem_CLS_vid(Dataset):
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
        
        self.ds = Load_vid_fair(self.directory, seq_length=self.seq_length, train=train)
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
    



class Belkacem_CLS_vit_balanced(Dataset):
    def __init__(self, directory_pos, directory_neg, seq_length=50, train=True, transform=None):
        """
        Args:
            directory (string): Directory with all the .json files of the positive and negative class (pos=belkacem)
        """
        self.directory_pos = directory_pos
        self.directory_neg = directory_neg
        self.seq_length = seq_length
        self.dt = 0.2
        self.transform = transform
        # # # Naive mode
        # self.videos = defaultdict(list)
        # # Fair mode
        self.videos = defaultdict(lambda: [int,defaultdict(list)])
        
        self.positive_samples = Load_vid_fair(self.directory_pos, seq_length=self.seq_length, train=train).videos
        self.negative_samples = Load_vid_fair(self.directory_neg, seq_length=self.seq_length, train=train).videos
        self.video_ids = list(self.positive_samples)



    def __len__(self):
        return len(list(self.positive_samples))

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        ## POSITIVE
        # Fair mode
        dict_frames_paths_pos = self.positive_samples[video_id][1]
        list_of_clip_pos = list(dict_frames_paths_pos)
        random_clip_id_pos = np.random.randint(0,len(list_of_clip_pos))
        frames_paths_pos = dict_frames_paths_pos[list_of_clip_pos[random_clip_id_pos]][0]
        vid_duration_pos = self.positive_samples[video_id][0]

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
        
        [pos_frames, pos_weight], [neg_frames, neg_weight], [anc_frames, anc_weight] = self.load_vid(frames_paths_pos, vid_duration_pos, self.transform), self.load_vid(frames_paths_neg, vid_duration_neg, self.transform), self.load_vid(frames_paths_anc, vid_duration_anc, self.transform)


        # Stack frames into a single tensor
        frames_tensor_pos, frames_tensor_neg = torch.stack(pos_frames), torch.stack(neg_frames)
        frames_tensor = torch.stack([frames_tensor_pos, frames_tensor_neg])
        # diff_frames_tensor = frames_tensor[:-1, :, :] - frames_tensor[1:, :, :]
        # diff_frames_tensor = torch.norm(diff_frames_tensor, dim=2, p=2)
        # Get label
        label = [np.int64(1), np.int64(0)]
        # # Center # and reduce each features?
        # frames_tensor = frames_tensor - torch.mean(frames_tensor, dim=0)
        # frames_tensor = torch.norm(frames_tensor - torch.mean(frames_tensor, dim=1)[:, None, :], dim=2)
        # diff_frames_tensor = frames_tensor[:, :-1, :, :, :] - frames_tensor[:, 1:, :, :, :]

        video_id_pos = video_id
        video_id_neg = list_of_neg[random_clip_id_neg]
        return frames_tensor, torch.tensor(label) , [video_id_pos, video_id_neg], torch.tensor([pos_weight, neg_weight])


    def load_vid(self, frames_paths, vid_duration, transform=None):
        frames = []
        # Load ldm file of the video
        # print(self.videos[video_id])
        cap = cv2.VideoCapture(frames_paths)
        nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sampling = 5 if np.abs(fps-25)<np.abs(fps-30) else 6
        # print(len(mat))
        # weight = vid_duration/ self.total_duration
        # Choose a random starting frame between first_frame and last_frame-length
        potential_starting_frame = range(int(nb_frame - ((self.seq_length)*sampling)*5))
        random_index = np.random.randint(0, len(potential_starting_frame))

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_index-1)
        count = 0
        first_frame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count == (self.seq_length)*sampling*5-1:
                break
            if (count % sampling == 0) and ret:
                if not first_frame:
                    previous_cropped_image = cropped_image
                # Initialize MTCNN for face detection
                mtcnn = MTCNN(keep_all=True)
                frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                # Detect faces
                boxes, _ = mtcnn.detect(frame_RGB)
                
                if boxes is not None:
                    first_frame = False
                    # Find the largest bounding box
                    largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                    x1, y1, x2, y2 = largest_box
                    if (min(int(y1-1),0) >= int(y2-1)) or (min(int(x1-1),0) >= int(x2-1)):
                        print(min(int(y1-1),0))
                        print(int(y2-1))
                        print(min(int(x1-1),0))
                        print(int(x2-1))

                    cropped_image = frame_RGB[max(int(y1-1),0):int(y2-1), max(int(x1-1),0):int(x2-1), :]

                if ((boxes is None) or len(boxes) == 0) and first_frame:
                    # If no frame found, redraw a starting frame
                    potential_starting_frame = range(int(nb_frame - ((self.seq_length)*sampling)*5))
                    random_index = np.random.randint(0, len(potential_starting_frame))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_index-1)
                    count = 0
                    cropped_image = previous_cropped_image
                    continue
                if cropped_image.shape[0] == 0:
                    print(f"Problem with video {frames_paths} at frame {random_index+count}.")
                    cropped_image = previous_cropped_image
                croppednresize = cv2.resize(cropped_image, (224,224))

                # print(f"No face found for video {frames_paths} at frame {random_index+count}.")
                frames.append(torch.tensor(croppednresize))
            count += 1
        del frame_RGB
        del croppednresize
        return frames, 1