import torch
import numpy as np


def torch_standardize_batch(batch: torch.Tensor):
    epsilon = 0.0000001
    batch_size, channels, _, _ = batch.shape
    im_mean = batch.view(batch_size, channels, -1).mean(2).view(batch_size, channels, 1, 1)
    # print(im_mean)
    im_std = batch.view(batch_size, channels, -1).std(2).view(batch_size, channels, 1, 1)
    # print(im_std)
    return (batch - im_mean) / (im_std + epsilon)


def torch_standardize_image(image: torch.Tensor, epsilon: float = 0.0000001):
    # USE move_mean_and_clip with matplotlib imshow to visualize the image
    channels, _, _ = image.shape
    im_mean = image.view(channels, -1).mean(1).view(channels, 1, 1)
    im_std = image.view(channels, -1).std(1).view(channels, 1, 1)
    return (image - im_mean) / (im_std + epsilon)


def np_standardize_image(image: np.array, epsilon: float = 0.0000001):
    _, _, channels = image.shape
    return (image - image.reshape((-1, channels)).mean(axis=0).reshape((1, 1, channels))) / (
            image.reshape((-1, channels)).std(axis=0) + epsilon).reshape((1, 1, channels))


def move_mean_and_clip(image: torch.Tensor, k: float = 0.7, t: float = 0.5):
    return torch.clip(image * k + t, min=0, max=1)
