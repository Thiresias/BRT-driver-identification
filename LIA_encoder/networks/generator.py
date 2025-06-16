from torch import nn
import torch
from .encoder import Encoder
from .styledecoder import Synthesis


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        # Uncomment for noIDleakage
        # print(f"==========> wa: {wa if wa is not None else 'None'}, alpha: {alpha if alpha is not None else 'None'}, feats: {feats if feats is not None else 'None'}")
        # print(f"==========> wa: {wa[0].shape if wa is not None else 'None'}, alpha: {alpha[0].shape if alpha is not None else 'None'}, feats: {feats[0].shape if feats is not None else 'None'}")
        motion = self.dec(wa, alpha, feats)

        # Uncomment for IDleakage
        # return alpha[0]

        # Uncomment for noIDleakage
        return motion
