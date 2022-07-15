import argparse
from datetime import datetime

import torch
from rich.console import Console
from torch.utils.data import Dataset

from autoencoder_dataset import OEImagesDatasetForEmbeddings, \
    OEImagesLabeledSetForEmbs
from utils.artifacts_util import small_ae_pretrain_model_artifact_saver
from utils.check_create_folder import check_create_folder
from utils.hooks import BottleneckEmbeddingExtractor
from models.autoencoder.autoencoder import AE
from models.autoencoder.autoencoder_model_functions import \
    embedding_production, set_model_parameters_for_embedding_creation

console = Console()


def params_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers",
                        '-w', type=int, default=4)
    parser.add_argument('--gpu_number',
                        '-g', type=int, default=0)
    parser.add_argument('--input_channels',
                        '-i', type=int, default=3)
    parser.add_argument('--root_path',
                        '-r', type=str, default=".")
    parser.add_argument('--model_path',
                        '-p', type=str)
    param = parser.parse_args()
    return param


def main():
    """
    if param.gpu_number = -1 it uses the cpu
    """
    param = params_parser()

    console.log(f'Using the following params:{param}')
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    image_size = (64, 64)
    if param.gpu_number == -1:
        cuda_gpu = f"cpu"
    else:
        cuda_gpu = f"cuda:{param.gpu_number}"
    device = torch.device(cuda_gpu if torch.cuda.is_available() else "cpu")
    (
        bottleneck,
        input_channel,
        layer_1_ft,
        widths,
        workers,
    ) = set_model_parameters_for_embedding_creation(param)
    # override bottleneck size for embedding production
    bottleneck = 128

    root_path = param.root_path
    model_path = param.model_path
    train_set_path = f"{root_path}/data/train_set"
    validation_set_path = f"{root_path}/data/validation_set"
    test_set_path = f"{root_path}/data/test_set"
    test_set_labels_csv = f"{root_path}/data/metadata/frames_labels.csv"
    outliers_set_path = f"{root_path}/data/outliers_set"
    outliers_set_labels_csv = f"{root_path}/data/metadata/outliers_frames_labels.csv"
    model_key = f"embeddings_{time_string}"
    model_save_folder = f"{root_path}/data/autoencoder/saves/{model_key}"

    model = AE(widths, image_shape=image_size, bottleneck_size=bottleneck)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    check_create_folder(model_save_folder)

    # Create embeddings
    train_set_for_embs = OEImagesDatasetForEmbeddings(
        train_set_path,
    )
    val_set_for_embs = OEImagesDatasetForEmbeddings(
        validation_set_path,
    )

    train_for_embs_loader = torch.utils.data.DataLoader(
        train_set_for_embs,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    val_for_embs_loader = torch.utils.data.DataLoader(
        val_set_for_embs,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    outliers_set_for_embs = OEImagesLabeledSetForEmbs(outliers_set_path, outliers_set_labels_csv)
    outliers_set_for_embs_loader = torch.utils.data.DataLoader(
        outliers_set_for_embs,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    test_set_for_embs = OEImagesLabeledSetForEmbs(test_set_path, test_set_labels_csv)
    test_for_embs_loader = torch.utils.data.DataLoader(
        test_set_for_embs,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    extractor_model = BottleneckEmbeddingExtractor(model)
    (
        train_embs_dict,
        oe_embs_dict,
        val_embs_dict,
        test_embs_dict
    ) = embedding_production(
        device,
        extractor_model,
        train_for_embs_loader,
        outliers_set_for_embs_loader,
        test_for_embs_loader,
        val_for_embs_loader,
    )
    console.log("Embeddings Creation Completed. Saving started")

    # SAVE STUFF
    artifacts_path = model_save_folder + "/artifacts/"
    small_ae_pretrain_model_artifact_saver(
        artifacts_path,
        train_embs_dict,
        oe_embs_dict,
        val_embs_dict,
        test_embs_dict,
    )
    console.log(f"Script completed, artifacts and embeddings located at {model_save_folder}.")


if __name__ == "__main__":
    main()
