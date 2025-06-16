import cv2
import torch


def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = tensor.clone()  # Avoid modifying the original tensor
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def display_batch(tensor):
    canvas = tensor[:, 0, :, :, :].permute(2,0,1)
    return canvas

def display_batch_lia(tensor):
    canvas = tensor[:, 0, :, :, :]
    return canvas