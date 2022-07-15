import pickle

from torch.utils.data import Dataset


class OEEmbeddingsDatasetOnlyAnomalies(Dataset):
    def __init__(
            self,
            file_path: str,
    ):
        with open(file_path, "rb") as pkf:
            data = pickle.load(pkf)
        self.embs = data["embeddings"]
        self.labels = data["labels"]

        an_ids = []
        for i, x in enumerate(self.labels):
            if x == 1:
                an_ids.append(i)
        self.an_ids = an_ids

    def __len__(self):
        return len(self.an_ids)

    def __getitem__(self, idx):
        set_id = self.an_ids[idx]
        return self.embs[set_id][0], self.labels[set_id]


class OEEmbeddingsDataset(Dataset):
    def __init__(
            self,
            file_path: str,
    ):
        with open(file_path, "rb") as pkf:
            data = pickle.load(pkf)
        self.embs = data["embeddings"]
        self.frame_paths = data["frame_paths"]

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        return self.embs[idx][0], self.frame_paths[idx]


class OEEmbeddingsOutliersSet(Dataset):
    def __init__(
            self,
            file_path: str,
            required_dataset_size:int,
    ):
        with open(file_path, "rb") as pkf:
            data = pickle.load(pkf)
        self.embs = data["embeddings"]
        self.frame_paths = data["frame_paths"]
        self.labels = data["labels"]
        all_idxs = [i for i in range(len(self.embs))]
        self.available_frames_idx = all_idxs
        if len(self.available_frames_idx) < required_dataset_size:
            #  if the available frames are less than the required frames the dataset is falsely increased
            self.len_dataset = required_dataset_size
        else:
            self.len_dataset = len(self.available_frames_idx)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        # this allows resampling since len_dataset could be larger that the real number of frames
        idx = idx % len(self.available_frames_idx)
        sel_id = self.available_frames_idx[idx]
        label = self.labels[sel_id]
        return self.embs[sel_id][0], label, self.frame_paths[sel_id]


class OEEmbeddingsTestSet(Dataset):
    def __init__(
            self,
            file_path: str,
    ):
        with open(file_path, "rb") as pkf:
            data = pickle.load(pkf)
        self.embs = data["embeddings"]
        self.frame_paths = data["frame_paths"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        emb = self.embs[idx][0]
        frame_path = self.frame_paths[idx]
        return emb, label, frame_path
