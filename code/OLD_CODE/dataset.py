import glob
import pickle
import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.feature_extraction.image import extract_patches_2d
from torch.utils.data import Dataset
from utils.image_util import torch_standardize_image


class UniformedPatchesDataset(Dataset):
    def __init__(self,
                 root_dir,
                 patch_shape,
                 max_patches: int,
                 aug_flag=False,
                 noise_flag=False,
                 transform=None,
                 noise_path=None,
                 noise_alpha: int = 60,
                 noise_p=0.5,
                 ):
        self.frames = sorted(glob.glob(root_dir + "/*"))
        self.aug_flag = aug_flag
        self.transform = transform
        self.noise_path = noise_path
        self.patch_shape = patch_shape

        self.noise_flag = noise_flag
        self.noise_alpha = noise_alpha
        self.noise_p = noise_p
        self.max_patches = max_patches
        if noise_flag:
            self.available_noises = len(glob.glob(noise_path + "/*"))
        else:
            self.available_noises = 0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):

        pt_image = torchvision.io.read_image(self.frames[idx]).numpy()
        image = np.transpose(pt_image, (1, 2, 0))
        # we apply augmentations and apply standardization.
        if self.aug_flag and self.transform is not None:
            aug_image = self.transform(image=image)["image"]
            if self.noise_flag and np.random.rand() <= self.noise_p:
                int_image = self.add_noise_to_image(aug_image)
            else:
                int_image = aug_image
        else:
            int_image = image
        torch__float = torch.from_numpy(np.transpose(int_image, (2, 0, 1))).float()
        numpy__float = torch_standardize_image(torch__float).numpy()
        final_image = np.transpose(numpy__float, (1, 2, 0))
        # then extract the patches
        if final_image.shape[:-1] == self.patch_shape:
            # we convert the patches to tensor
            patches = torch.tensor(final_image).unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        else:
            np_patch = extract_patches_2d(final_image, self.patch_shape, max_patches=self.max_patches)
            # we convert the patches to tensor
            patches = torch.tensor(np_patch).permute(0, 3, 1, 2).contiguous()

        return patches

    def add_noise_to_image(self, aug_image):
        noise_id = np.random.randint(self.available_noises)
        with open(self.noise_path + f"/noise_{noise_id}.pk", "rb") as pk_file:
            noise = pickle.load(pk_file)
        x_crop = np.random.randint(1000 - aug_image.shape[1])
        y_crop = np.random.randint(1000 - aug_image.shape[0])
        crop_noise = A.Crop(
            x_min=x_crop,
            y_min=y_crop,
            x_max=x_crop + aug_image.shape[1],
            y_max=y_crop + aug_image.shape[0],
            always_apply=True, p=1.0,
        )(image=noise)["image"]
        noise_rand_alpha = np.random.randint(low=self.noise_alpha, high=100) / 100.0
        final_image = aug_image / 255 + (crop_noise * (1.0 - noise_rand_alpha))
        return final_image


class UniformedPatchesTestset(Dataset):
    def __init__(
            self,
            root_dir: str,
            patch_shape,
            label_csv: str,
            max_patches: int = 128,
    ):
        self.frames = sorted(glob.glob(root_dir + "/*"))
        self.labels = pd.read_csv(label_csv)
        self.max_patches = max_patches
        self.patch_shape = patch_shape

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # we read a pil image
        pt_image = torch_standardize_image(torchvision.io.read_image(self.frames[idx]).float()).numpy()
        final_image = np.transpose(pt_image, (1, 2, 0))
        # then extract the patches
        if final_image.shape[:-1] == self.patch_shape:
            # we convert the patches to tensor
            patches = torch.tensor(final_image).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        else:
            np_patch = extract_patches_2d(final_image, self.patch_shape, max_patches=self.max_patches)
            # we convert the patches to tensor
            patches = torch.tensor(np_patch).permute(0, 3, 1, 2).contiguous()
        sample = {'patches': patches, 'label': self.labels["label"].values[idx], "frame_id": idx}
        return sample


class UniformedPatchesDatasetAndPath(Dataset):
    def __init__(self, root_dir, limit=False):
        if limit:
            self.frames = glob.glob(root_dir + "/*")[:3000]
        else:
            self.frames = sorted(glob.glob(root_dir + "/*"))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.frames[idx]).float()

        return image, self.frames[idx]
