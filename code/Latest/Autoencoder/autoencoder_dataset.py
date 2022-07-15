import glob
import pickle
from typing import List, Tuple, Dict

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision
from rich.traceback import install
from torch.utils.data import Dataset

from utils.image_util import torch_standardize_image

install()


class OEImagesDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 image_shape: Tuple[int, int],
                 aug_flag=False,
                 noise_flag=False,
                 transform=None,
                 noise_path=None,
                 noise_alpha: int = 60,
                 noise_p=0.5,
                 ):
        list_files = sorted(glob.glob(root_dir + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.frames = list_files
        self.aug_flag = aug_flag
        self.transform = transform
        self.noise_path = noise_path
        self.image_shape = image_shape
        self.noise_flag = noise_flag
        self.noise_alpha = noise_alpha
        self.noise_p = noise_p
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
        torch_float = torch.from_numpy(np.transpose(int_image, (2, 0, 1))).float()
        final_image = torch_standardize_image(torch_float).contiguous()
        return final_image

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


class OEImagesDatasetForEmbeddings(Dataset):
    def __init__(self,
                 root_dir: str
                 ):
        list_files = sorted(glob.glob(root_dir + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.frames = list_files

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        final_image = torch_standardize_image(torchvision.io.read_image(frame_path).float()).contiguous()
        return final_image, frame_path


class OEImagesTestset(Dataset):
    def __init__(
            self,
            test_set_path: str,
            label_csv_path: str
    ):
        list_files = sorted(glob.glob(test_set_path + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.label_df = pd.read_csv(label_csv_path)["label"].values
        self.frames = list_files

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # we read a pil image
        sel_id = idx
        # sel_id = self.frames[idx]
        final_image = torch_standardize_image(torchvision.io.read_image(self.frames[sel_id]).float()).contiguous()
        return final_image, self.label_df[sel_id]


class OEImagesLabeledSetForEmbs(Dataset):
    def __init__(
            self,
            root_dir: str,
            csv_path: str
    ):
        list_files = sorted(glob.glob(root_dir + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.label_df = pd.read_csv(csv_path)["label"].values
        self.frames = list_files

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # we read a pil image
        # sel_id = self.frames[idx]
        sel_id = idx
        final_image = torch_standardize_image(torchvision.io.read_image(self.frames[sel_id]).float()).contiguous()
        sample = {'image': final_image, 'label': self.label_df[sel_id], "frame_path": self.frames[sel_id]}
        return sample


class OEImagesTestsetForEmbs(Dataset):
    def __init__(
            self,
            dict_root_dir_label_csv: Dict[str, str],
            dataset_ids: List[int],
            oe_flag=False,
    ):
        """
        produces all labels since it is for embeddings production
        Args:
            dict_root_dir_label_csv:
            oe_flag: if true produces only anomalies
        """
        self.frames = []
        self.labels = []
        self.difficulty = []

        for dataset_id, (root_dir, label_csv) in zip(dataset_ids, dict_root_dir_label_csv.items()):
            list_files = sorted(glob.glob(root_dir + "/*"))
            assert len(list_files) != 0, "Error in loading frames"
            self.frames = self.frames + list_files
            df = pd.read_csv(label_csv)
            self.labels = self.labels + [f"{dataset_id}-{el}" for el in df["label"].tolist()]
            self.difficulty = self.difficulty + df["difficulty"].tolist()

        if oe_flag:
            self.available_frames_idx = [i for i in range(len(self.frames)) if self.labels[i].split('-')[1] != '0']
        else:
            self.available_frames_idx = [i for i in range(len(self.frames))]

    def __len__(self):
        return len(self.available_frames_idx)

    def __getitem__(self, idx):
        # we read a pil image
        sel_id = self.available_frames_idx[idx]
        frame_path = self.frames[sel_id]
        final_image = torch_standardize_image(torchvision.io.read_image(frame_path).float()).contiguous()
        sample = {'image': final_image, 'label': self.labels[sel_id], "frame_path": frame_path,
                  'difficulty': self.difficulty[sel_id]}
        return sample
