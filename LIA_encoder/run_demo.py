import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import cv2
from collections import defaultdict
from networks.classifier import LSTMClassifier

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


# def vid_preprocessing(vid_path):
#     vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
#     vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
#     fps = vid_dict[2]['video_fps']
#     vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

#     return vid_norm, fps

def vid_preprocessing(vid_path, size=256):
    # vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    cap = cv2.VideoCapture(vid_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    print(video.shape)
    video = np.array([
        cv2.resize(frame, (size, size)) for frame in video
    ])

    vid = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)
    fps = video_fps
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps

def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


def save_motion(motion, save_folder_path):
    save_path = save_folder_path+'test.json'
    with open(save_path, 'w') as f:
        json.dump(motion, f)


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        if args.model == 'vox':
            model_path = 'checkpoints/vox.pt'
        elif args.model == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif args.model == 'ted':
            model_path = 'checkpoints/ted.pt'
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.save_path = args.save_folder + '/%s' % args.model
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path = os.path.join(self.save_path, Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.mp4')
        self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        # self.vid_target, self.fps = vid_preprocessing(args.driving_path)
        self.vid_target, self.fps = vid_preprocessing(args.driving_path, args.size)
        self.vid_target = self.vid_target.cuda()

    def run(self):

        print('==> running')
        with torch.no_grad():

            vid_motion = []
            vid_dict = defaultdict()

            if self.args.model == 'ted':
                h_start = None
            else:
                h_start = self.gen.enc.enc_motion(self.vid_target[:, 0, :, :, :])

            for i in tqdm(range(self.vid_target.size(1))):
                img_target = self.vid_target[:, i, :, :, :]
                motion_vector = self.gen(self.img_source, img_target, h_start)

                vid_motion.append(motion_vector.unsqueeze(0))
                vid_dict[f"frame_{i*6}"] = motion_vector.unsqueeze(0).tolist()

            # vid_target_recon = torch.cat(vid_target_recon, dim=2)
            save_motion(vid_dict, self.save_path)


def train(args):
    if args.model == 'vox':
        model_path = 'checkpoints/vox.pt'
    print('==> Load Model')
    gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
    weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
    gen.load_state_dict(weight)
    gen.eval()

    print('==> loading data')
    save_path = args.save_folder + '/%s' % args.model
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.mp4')
    img_source = img_preprocessing(args.source_path, args.size).cuda()
    # self.vid_target, self.fps = vid_preprocessing(args.driving_path)
    vid_target, fps = vid_preprocessing(args.driving_path, args.size)
    vid_target = vid_target.cuda()


    print('==> running')
    with torch.no_grad():

        vid_motion = []
        vid_dict = defaultdict()

        if args.model == 'ted':
            h_start = None
        else:
            h_start = gen.enc.enc_motion(vid_target[:, 0, :, :, :])

        for i in tqdm(range(vid_target.size(1))):
            img_target = vid_target[:, i, :, :, :]
            motion_vector = gen(img_source, img_target, h_start)

            

        # vid_target_recon = torch.cat(vid_target_recon, dim=2)
        save_motion(vid_dict, save_path)


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_folder", type=str, default='./res')
    args = parser.parse_args()

    # Load data 
    # demo
    demo = Demo(args)
    demo.run()
