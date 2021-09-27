import argparse
from collections import defaultdict
from typing import Tuple

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from paper_code_release.model.autoencoder import AE
from paper_code_release.paper_utils.losses_util import losses_list, compute_losses
from paper_code_release.paper_utils.variables_util import available_scale_levels, dataset_names, training_patches


def set_model_and_train_parameters_patches(
        device: torch.device,
        ml_flow_run_id: str,
        param: argparse.ArgumentParser,
        image_shape: Tuple[int, int],
):
    batch_size = param.batch_size
    lr = param.learning_rate
    epochs = param.max_epochs
    max_patience = int(np.log2(epochs)) + 2
    patience_threshold = param.patience_thres
    workers = param.num_workers
    input_channel = param.input_channels
    bottleneck = param.bottleneck
    layer_1_ft = param.first_layer_size
    layer_2_ft = layer_1_ft * 2
    layer_3_ft = layer_1_ft * 2 * 2
    layer_4_ft = layer_1_ft * 2 * 2 * 2
    widths_ = [
        input_channel,
        layer_1_ft,
        layer_2_ft,
        layer_3_ft,
        layer_4_ft,
    ]
    if param.use_ml_flow:
        ml_flow_run_id = mlflow.active_run().info.run_id
        mlflow.log_param("image_shape", image_shape)
        mlflow.log_param("max_epochs", param.max_epochs)
        mlflow.log_param("batch_size", param.batch_size)
        mlflow.log_param("bottleneck", param.bottleneck)
        mlflow.log_param("num_workers", param.num_workers)
        mlflow.log_param("learning_rate", param.learning_rate)
        mlflow.log_param("first_layer_size", param.first_layer_size)
        mlflow.log_param("input_channels", param.input_channels)
        mlflow.log_param("optimized_loss", losses_list[param.id_optimized_loss])
        mlflow.log_param("device", device)
        mlflow.log_param("max_patience", max_patience)
        mlflow.log_param("patience_threshold", patience_threshold)
        mlflow.log_param("scale_level", available_scale_levels[param.scale_level])
        mlflow.log_param("dataset", dataset_names[param.dataset])
        mlflow.log_param("train_patches_number", training_patches[param.scale_level])
        mlflow.log_param("test_patches_number", param.test_patches_number)
    return batch_size, bottleneck, epochs, input_channel, layer_1_ft, lr, max_patience, ml_flow_run_id, widths_, workers


def train_loop_patches(
        model: AE,
        loader: torch.utils.data.DataLoader,
        optimizer,
        device: torch.device,
        id_loss: int,
):
    running_mse = []
    running_mae = []
    model.train()
    clip_value = 10
    epoch_norm = 0
    for data in loader:
        # data has a shape of (batchsize//patchsize, patchsize, channels, height, width)
        # we change is to (batchsize,image)
        data_shape = data.shape
        data = data.view(data_shape[0] * data_shape[1], data_shape[2], data_shape[3], data_shape[4])
        input_data = data.to(device)

        optimizer.zero_grad()
        output_sample = model(input_data)
        losses = compute_losses(input_data, output_sample)
        loss = losses[id_loss]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        epoch_norm = total_norm ** 0.5

        optimizer.step()

        running_mse.append(losses[0].item())
        running_mae.append(losses[1].item())

    epoch_mse = np.mean(running_mse)
    epoch_mae = np.mean(running_mae)

    return epoch_mse, epoch_mae, epoch_norm


def val_loop_patches(
        model: AE,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
):
    running_mse = []
    running_mae = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            # for data in tqdm(loader, total=len(loader)):
            data_shape = data.shape
            data = data.view(data_shape[0] * data_shape[1], data_shape[2], data_shape[3], data_shape[4])
            input_data = data.to(device)

            output_sample = model(input_data)
            losses = compute_losses(input_data, output_sample)
            running_mse.append(losses[0].item())
            running_mae.append(losses[1].item())

    epoch_mse = np.mean(running_mse)
    epoch_mae = np.mean(running_mae)

    return epoch_mse, epoch_mae


def epoch_loop_patches(
        device: torch.device,
        epochs: int,
        model: AE,
        optimizer,
        param: argparse.ArgumentParser,
        scheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
):
    # Epoch Loop
    etqdm = tqdm(range(epochs), total=epochs, postfix="Training")
    metrics = defaultdict(list)
    total_epoch_norms = 0
    count_ep = 0
    for epoch in etqdm:
        count_ep += 1
        train_loop_returns = train_loop_patches(
            model,
            train_loader,
            optimizer,
            device,
            param.id_optimized_loss,
        )
        total_epoch_norms += train_loop_returns[2]
        last_avg_norm = total_epoch_norms / count_ep

        if param.use_ml_flow:
            mlflow.log_metric(f'train_{losses_list[0]}', train_loop_returns[0], epoch)
            mlflow.log_metric(f'train_{losses_list[1]}', train_loop_returns[1], epoch)
            mlflow.log_metric(f'train_epoch_norm', train_loop_returns[2], epoch)
            mlflow.log_metric(f'last_avg_ep_norm_until_now', last_avg_norm, epoch)

        metrics[f'train_{losses_list[0]}'].append(train_loop_returns[0])
        metrics[f'train_{losses_list[1]}'].append(train_loop_returns[1])

        val_losses = val_loop_patches(model, val_loader, device)
        scheduler.step(val_losses[param.id_optimized_loss])

        if param.use_ml_flow:
            mlflow.log_metric(f'val_{losses_list[0]}', val_losses[0], epoch)
            mlflow.log_metric(f'val_{losses_list[1]}', val_losses[1], epoch)
        metrics[f'val_{losses_list[0]}'].append(val_losses[0])
        metrics[f'val_{losses_list[1]}'].append(val_losses[1])

        etqdm.set_postfix({f'Train {losses_list[param.id_optimized_loss]}': train_loop_returns[param.id_optimized_loss],
                           f'Val {losses_list[param.id_optimized_loss]}': val_losses[param.id_optimized_loss]})

    return metrics


def test_loop_df_rows(device: torch.device, model: AE, test_loader: torch.utils.data.DataLoader):
    model.eval()

    df_dict = {'frame_id': [], 'patch_id': [], 'frame_label': [], 'mse_loss': [], 'mae_loss': []}
    # DEBUG
    # count = 0
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), postfix="Running Test Set inference"):
            # data contains patches coming from a single frame so all with the same label
            # we do the inference
            inputs_ = data['patches'][0].to(device)
            output_data = model(inputs_)
            for i, (y_true, y_pred) in enumerate(zip(inputs_, output_data)):
                losses = compute_losses(y_true, y_pred)
                df_dict["frame_id"].append(data["frame_id"].item())
                df_dict["patch_id"].append(i)
                df_dict["frame_label"].append(data["label"].item())
                df_dict["mse_loss"].append(losses[0].item())
                df_dict["mae_loss"].append(losses[1].item())
            # DEBUG
            # if count == 25:
            #     return df_dict
            # count += 1
    return df_dict
