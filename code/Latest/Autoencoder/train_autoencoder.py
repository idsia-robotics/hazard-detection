import argparse
from datetime import datetime

import albumentations as A
import torch
from rich.console import Console
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from autoencoder_dataset import OEImagesDataset, OEImagesTestset
from utils.artifacts_util import ae_pretrain_model_artifact_saver
from utils.check_create_folder import check_create_folder
from models.autoencoder.autoencoder import AE
from models.autoencoder.autoencoder_model_functions import epoch_loop_ae_oe, test_loop_oe, \
    set_model_and_train_parameters_autoencoder, autoencoder_auc_computation_and_logging

console = Console()


def params_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        "-b", type=int, default=320)
    parser.add_argument("--bottleneck",
                        "-n", type=int, default=128)
    parser.add_argument("--num_workers",
                        '-w', type=int, default=4)
    parser.add_argument('--gpu_number',
                        '-g', type=int, default=0)
    parser.add_argument('--input_channels',
                        '-i', type=int, default=3)
    parser.add_argument('--root_path',
                        '-r', type=str, default=".")
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
        batch_size,
        bottleneck,
        epochs,
        input_channel,
        layer_1_ft,
        lr,
        max_patience,
        widths,
        workers,
    ) = set_model_and_train_parameters_autoencoder(param)
    root_path = param.root_path
    noise_path = f"{root_path}/data/perlin_noise"
    train_set_path = f"{root_path}/data/train_set"
    validation_set_path = f"{root_path}/data/validation_set"
    test_set_path = f"{root_path}/data/test_set"
    test_set_labels_csv = f"{root_path}/data/metadata/frames_labels.csv"
    model_key = f"AE_B{bottleneck}F{layer_1_ft}_{time_string}"
    model_save_folder = f"{root_path}/data/autoencoder/saves/{model_key}"
    check_create_folder(model_save_folder)

    model = AE(widths, image_shape=image_size, bottleneck_size=bottleneck)
    model.to(device)
    # DATA INIT
    composed_transform = A.Compose(
        [
            A.transforms.HorizontalFlip(p=0.5),
            A.transforms.RandomBrightnessContrast(brightness_limit=0.1,
                                                  contrast_limit=0.1,
                                                  p=0.5),
            A.RandomSizedCrop(min_max_height=[50, image_size[0]],
                              height=image_size[0],
                              width=image_size[1],
                              p=0.5),
            A.Rotate(limit=10,
                     p=0.5),
        ]
    )
    train_set = OEImagesDataset(
        train_set_path,
        image_shape=image_size,
        aug_flag=True,
        noise_flag=True,
        transform=composed_transform,
        noise_path=noise_path,
    )
    val_set = OEImagesDataset(validation_set_path,
                              image_shape=image_size,
                              aug_flag=False,
                              noise_flag=False,
                              )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        train_sample_batch = next(iter(train_loader))

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=max_patience, verbose=True)

    # TRAIN
    metrics = epoch_loop_ae_oe(device, epochs, model, optimizer, scheduler, train_loader, val_loader)

    # TEST
    console.log("Training Completed, Testing Started")
    test_set = OEImagesTestset(test_set_path, test_set_labels_csv)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    with torch.no_grad():
        test_sample_batch = next(iter(test_loader))

    test_losses, test_labels = test_loop_oe(device, model, test_loader)

    # Compute AUC for ml flow logging
    autoencoder_auc_computation_and_logging(test_labels, test_losses)
    console.log("Testing Completed. Embedding Creation Started")

    # SAVE STUFF
    check_create_folder(model_save_folder)
    console.log("Saving model and artifacts")
    artifacts_path = model_save_folder + "/artifacts/"
    ae_pretrain_model_artifact_saver(
        batch_size,
        bottleneck,
        epochs,
        layer_1_ft,
        lr,
        metrics,
        model,
        input_channel,
        image_size,
        artifacts_path,
        param=param,
        bfs_key=model_key,
        train_sample_batch=train_sample_batch,
        test_sample_batch=test_sample_batch,
    )
    console.log(f"Script completed! Model and artifacts located at:\n{model_save_folder}")


if __name__ == "__main__":
    main()
