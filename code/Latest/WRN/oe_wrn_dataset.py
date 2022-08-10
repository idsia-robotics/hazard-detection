import glob
import pandas as pd
import torchvision
from rich.traceback import install
from torch.utils.data import Dataset

install()


class OEWRNImagesDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform,
                 env_labels_csv: str,
                 ):
        list_files = sorted(glob.glob(root_dir + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.frames = list_files
        self.transform = transform
        self.labels = pd.read_csv(env_labels_csv)["env"].values

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        pt_image = torchvision.io.read_image(self.frames[idx]).float() / 255.0
        final_image = self.transform(pt_image)
        return final_image, self.labels[idx]


class OEWRNImagesOutlierset(Dataset):
    def __init__(
            self,
            files_path: str,
            label_csv_path: str,
            transform,
            required_dataset_size=0,
    ):
        list_files = sorted(glob.glob(files_path + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.label_df = pd.read_csv(label_csv_path)["label"].values
        self.frames = list_files
        self.transform = transform
        all_idxs = [i for i in range(len(self.frames))]
        self.available_frames_idx = all_idxs

        if len(self.available_frames_idx) < required_dataset_size:
            self.len_dataset = required_dataset_size
        else:
            self.len_dataset = len(self.available_frames_idx)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        idx = idx % len(self.available_frames_idx)
        sel_id = self.available_frames_idx[idx]
        pt_image = torchvision.io.read_image(self.frames[idx]).float() / 255.0
        final_image = self.transform(pt_image)
        return final_image, self.label_df[sel_id]


class OEWRNImagesTestset(Dataset):
    def __init__(
            self,
            test_set_path: str,
            label_csv_path: str,
            transform,
    ):
        list_files = sorted(glob.glob(test_set_path + "/*"))
        assert len(list_files) != 0, "Error in loading frames"
        self.labels = pd.read_csv(label_csv_path)["label"].values
        self.frames = list_files
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # we read a pil image
        sel_id = idx
        pt_image = torchvision.io.read_image(self.frames[idx]).float() / 255.0
        final_image = self.transform(pt_image)
        return final_image, self.labels[sel_id]
