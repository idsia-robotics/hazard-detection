import pickle
from random import shuffle

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
            number_of_outliers=None,  # only used for analysis 2
            selected_anomalies=None,
            analysis_version=0,
            required_dataset_size=0
            # This is the size that the dataset should have to fit the normal data sampling and batching
    ):
        with open(file_path, "rb") as pkf:
            data = pickle.load(pkf)
        self.embs = data["embeddings"]
        self.frame_paths = data["frame_paths"]
        self.labels = data["labels"]
        all_idxs = [i for i in range(len(self.embs))]
        if analysis_version in [1, 4, 5]:  # use all Outlier set
            self.available_frames_idx = all_idxs

        elif analysis_version == 2:  # use a subset of the Outlier set
            assert number_of_outliers, "Analysis 2 requires a set number of Outliers"
            shuffle(all_idxs)
            self.available_frames_idx = all_idxs[:number_of_outliers]

        elif analysis_version == 3:  # use a subset of the anomaly type availables
            assert selected_anomalies, "Analysis 3 requires the prior definition of the anomalies to be used"
            self.selected_anomalies = selected_anomalies
            self.available_frames_idx = [i for i in all_idxs
                                         if self.labels[i] in selected_anomalies]

        else:
            assert analysis_version in [1, 2, 3, 4, 5], "Analysis code has to be 1, 2, 3, 4, 5"

        if len(self.available_frames_idx) < required_dataset_size:
            # this is useful only for analysis 2 when the available frame are less than those required
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

#
# # keeping old class for reference
# class Old_OEEmbeddingsTestset(Dataset):
#     def __init__(
#             self,
#             file_path: str,
#             only_obvious_anomalies: int,  # will ignore for now since it will alwasys be 1
#             number_oe_anomalies=None,
#             predefined_available_frames=None,
#             oe_flag=False,
#             analysis_version=0,
#             num_oe_sample=0,
#     ):
#         with open(file_path, "rb") as pkf:
#             data = pickle.load(pkf)
#         self.embs = data["embeddings"]
#         self.frame_paths = data["frame_paths"]
#         self.labels = [labels_to_combined_labels[el] for el in data["labels"]]
#         self.difficulty = data["difficulty"]
#         if predefined_available_frames is None:
#             self.available_frames_idx = [i for i in range(len(self.embs))]
#         else:
#             self.available_frames_idx = predefined_available_frames
#         self.only_obvious_anomalies = only_obvious_anomalies
#         self.oe_flag = oe_flag
#         if only_obvious_anomalies:
#             self.available_frames_idx = [i for i in self.available_frames_idx if self.difficulty[i] != 1]
#         if oe_flag:
#             self.available_frames_idx = [i for i in self.available_frames_idx if self.labels[i] != 0]
#
#         if analysis_version == 2:
#             shuffle(self.available_frames_idx)
#             self.available_frames_idx = self.available_frames_idx[:num_oe_sample]
#         self.dataset_len = len(self.available_frames_idx)
#         # dataset len is set before specific_oe_anomalies selection to allow a resampling
#         self.selected_anomalies = None
#         if oe_flag and number_oe_anomalies is not None and analysis_version == 3:  # BUG we are taking in 0s
#             avail_anomalies = list(combined_labels_to_names.keys())[1:]  # here we exclude 0
#             shuffle(avail_anomalies)
#             selected_anomalies = avail_anomalies[:number_oe_anomalies]
#             self.selected_anomalies = selected_anomalies
#             self.available_frames_idx = [i for i in self.available_frames_idx
#                                          if self.labels[i] in selected_anomalies]
#
#     def __len__(self):
#         return self.dataset_len
#
#     def __getitem__(self, idx):
#         # this allows resampling in case of small amount of samples selected since self.dataset_len >= len(self.avalilable_frames_idx)
#         idx = idx % len(self.available_frames_idx)
#         sel_id = self.available_frames_idx[idx]
#         emb_difficulty = self.difficulty[sel_id]
#         label = self.labels[sel_id]
#
#         if self.only_obvious_anomalies:
#             assert emb_difficulty in [0,
#                                       2], "Something is wrong with the OEEmbeddingsTestset class and tricky anomalies are passing"
#         if self.oe_flag:
#             assert label != 0, f"Something is wrong with the OEEmbeddingsTestset class and normal samples are passing {label}"
#         return self.embs[sel_id][0], label, emb_difficulty, self.frame_paths[sel_id]
