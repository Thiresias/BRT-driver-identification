import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
# from augmentations import AugmentationTransform
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Vox256(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/vox/train'
        elif split == 'test':
            self.ds_path = './datasets/vox/test'
        else:
            raise NotImplementedError

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

            return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Vox256_vox2german(Dataset):
    def __init__(self, transform=None):
        self.source_root = './datasets/german/'
        self.driving_root = './datasets/vox/test/'

        self.anno = pd.read_csv('pairs_annotations/german_vox.csv')

        self.source_imgs = os.listdir(self.source_root)
        self.transform = transform

    def __getitem__(self, idx):
        source_name = str('%03d' % self.anno['source'][idx])
        driving_name = self.anno['driving'][idx]

        source_vid_path = self.source_root + source_name
        driving_vid_path = self.driving_root + driving_name

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.source_imgs)


class Vox256_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Vox256_cross(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.anno = pd.read_csv('pairs_annotations/vox256.csv')
        self.transform = transform

    def __getitem__(self, idx):
        source_name = self.anno['source'][idx]
        driving_name = self.anno['driving'][idx]

        source_vid_path = os.path.join(self.ds_path, source_name)
        driving_vid_path = os.path.join(self.ds_path, driving_name)

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.videos)


class Taichi(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/taichi/train/'
        else:
            self.ds_path = './datasets/taichi/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(True, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):

        video_path = self.ds_path + self.videos[idx]
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Taichi_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/taichi/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class TED(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/ted/train/'
        else:
            self.ds_path = './datasets/ted/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class TED_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/ted/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


import cv2
import json
import torch
from collections import defaultdict

class MvtAnalysis(Dataset):
    def __init__(self, directory, train=False):
        """
        Args:
            directory (string): Directory with all the .mat files.
        """
        self.directory = directory
        self.videos = defaultdict(list)

        # Iterate through the directory and group videos
        for filename in os.listdir(self.directory):
            if filename.endswith('.mp4'):
                vidnum = int(filename.split('_')[1][:4])
                if train and vidnum<8:
                    self.videos[filename[:-4]] = os.path.join(self.directory, filename)
                elif not train and vidnum>=8:
                    self.videos[filename[:-4]] = os.path.join(self.directory, filename)

        # Drop if missing frames (not equal 40)
        for video in list(self.videos):
            vid_path = self.videos[video]
            cap = cv2.VideoCapture(vid_path)
            video_nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if video_nb_frame < (5 if np.abs(fps-25)<np.abs(fps-30) else 6)*25:
                print(f"Clip {video} is less than a 5sec long video.")
                del self.videos[video]

        self.video_ids = list(self.videos)
        print(self.video_ids)
    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_path = self.videos[video_id]
        frames = vid_preprocessing(video_path)[0]

        # Get label
        vid_name = os.path.basename(video_path) # video_path: path/to/idX_vidN.mp4
        id = vid_name.split('_')[0]     # vid_name: idX_vidN.mp4
        num = int(id[2:])               # id: idX
        label = CDFlabel_transfo(num)   # num: X'
        label = int(label)

        # # Apply median filter across the temporal dimension (T, n)
        # kernel_size = 7
        # frames_tensor_AU_np = frames_tensor_AU.numpy()  # Convert to numpy for processing
        # frames_tensor_AU_np = medfilt(frames_tensor_AU_np, kernel_size=[kernel_size, 1])
        # frames_tensor_AU = torch.tensor(frames_tensor_AU_np)  # Convert back to torch tensor
        
        
        
        return frames, label, video_id
    
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


import numpy as np
def vid_preprocessing(vid_path, size=256):
    # vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    cap = cv2.VideoCapture(vid_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = 5 if np.abs(video_fps-25)<np.abs(video_fps-30) else 6 # For 25fps and 30fps videos, get one frame every 0.2s
    video = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret and count%sample==0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.append(frame)
            
        elif not ret:
            break
        elif count//sample==39:
            break
        count += 1
        
    cap.release()
    video = np.array([
        cv2.resize(frame, (size, size)) for frame in video
    ])
    vid = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)
    fps = video_fps
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]
    return vid_norm, fps





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
            if obama and filename in ['179.mp4', '183.mp4', '826.mp4']:
                continue
            if filename.endswith('.mp4'):
                video_id = "".join(filename.split('_')[0]) if not (('FF' in filename) or ('id' in filename))else f'{filename}'
                clip_name = filename
                cap = cv2.VideoCapture(os.path.join(self.directory, filename))
                nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = nb_frame/fps
                # if fps >20:
                #     if duration >= seq_length:
                #         self.total_duration += duration
                #         if video_id in list(self.videos):
                #             self.videos[video_id][0] = self.videos[video_id][0] + duration
                #             self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                #         else:
                #             self.videos[video_id][0] = duration
                #             self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                    
                #     else:
                #         print(f"Clip {clip_name} duration is {duration} sec (<{seq_length} sec).")
                # else:
                #     print(f'Not enough fps (found {fps} need at least 20)')

                if (fps == 30) or (fps == 25) or (np.abs(fps - 30)<0.1 or np.abs(fps-25)<0.1):
                    if duration >= seq_length:
                        self.total_duration += duration
                        if video_id in list(self.videos):
                            self.videos[video_id][0] = self.videos[video_id][0] + duration
                            self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                        else:
                            self.videos[video_id][0] = duration
                            self.videos[video_id][1][clip_name].append(os.path.join(self.directory, filename))
                    
                    else:
                        print(f"Clip {clip_name} is less than {seq_length} sec long.")
                else:
                    print(f"Clip {clip_name} has a {fps} fps. Need to be 25 or 30.")

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
    def __init__(self, directory_pos, directory_neg, transform=None, seq_length=50, train=True, obama=False):
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
        
        self.positive_samples = Load_vid_fair(self.directory_pos, seq_length=self.seq_length, train=train, pos=True, obama=obama).videos
        self.negative_samples = Load_vid_fair(self.directory_neg, seq_length=self.seq_length, train=train, pos=False, obama=obama).videos
        for vidname in list(self.positive_samples):
            self.videos[vidname] = self.positive_samples[vidname]
        for vidname in list(self.negative_samples):
            self.videos[vidname] = self.negative_samples[vidname]
        self.video_ids = list(self.videos)
        self.K = 10
        print(self.videos)

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

            [sequence, fps] = self.load_vid(frames_paths, vid_duration, self.transform) # BRT-DFD baseline
            # [sequence, fps] = self.load_vid_multiscale(frames_paths, vid_duration, self.transform) # BRT-DFD+
            # label = 0 if (('FF' in video_id) or ('id' in video_id)) else 1
            label = 0 if 'drivingCDF' in frames_paths else 1

            frames.append(sequence)
            labels.append(label)
            video_ids.append(video_id)
            fps_list.append(fps)
        np.random.seed(0)
        
        frames_tensor = torch.stack(frames)
        label_tensor = torch.tensor(labels)
        fps_tensor = torch.tensor(fps_list)

        # # Center # and reduce each features?
        # frames_tensor = frames_tensor - torch.mean(frames_tensor, dim=0)
        # frames_tensor = torch.norm(frames_tensor - torch.mean(frames_tensor, dim=1)[:, None, :], dim=2)
        # diff_frames_tensor = frames_tensor[:, :-1, :, :, :] - frames_tensor[:, 1:, :, :, :]


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
        # Load video
        cap = cv2.VideoCapture(frames_paths)
        nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sampling = 5 if np.abs(fps-25)<np.abs(fps-30) else 6
        
        # Choose a random starting frame between first_frame and last_frame-length
        random_interval = np.random.randint(1,5)
        potential_starting_frame = range(int(nb_frame - self.seq_length*fps ))
        random_index = np.random.randint(0, len(potential_starting_frame)+1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_index-1)
        count = 0
        accepted_frames = 0
        MAX_FRAME = 40
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
    



class Belkacem_CLS_lia_balanced(Dataset):
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
        # frames_tensor_pos, frames_tensor_neg = torch.stack(pos_frames), torch.stack(neg_frames)
        frames_tensor = torch.stack([pos_frames, neg_frames])
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
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count == (self.seq_length)*sampling*5-1:
                break
            if (count % sampling == 0) and ret:
                frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(torch.tensor(frame_RGB))
            count += 1
        del frame_RGB

        cap.release()
        vid = torch.stack(frames).permute(0, 3, 1, 2)
        # vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]
        return vid, 1