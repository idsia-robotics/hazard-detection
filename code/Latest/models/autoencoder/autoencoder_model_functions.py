import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from utils.losses_util import losses_list, compute_losses
from utils.metrics_util import compute_pr_aucs, compute_roc_aucs
from models.autoencoder.autoencoder import AE


# USED
def set_model_and_train_parameters_autoencoder(
        param: argparse.ArgumentParser,
):
    batch_size = param.batch_size
    lr = 0.001
    epochs = 100
    max_patience = int(np.log2(epochs)) + 2
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
    return batch_size, bottleneck, epochs, input_channel, layer_1_ft, lr, max_patience, widths_, workers


# USED
def set_model_parameters_for_embedding_creation(
        param: argparse.ArgumentParser,
):
    workers = param.num_workers
    input_channel = param.input_channels
    bottleneck = 128
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
    return bottleneck, input_channel, layer_1_ft, widths_, workers


# USED
def train_loop_oe(
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
    # DEBUG
    # counter = 0
    for data in loader:
        images = data
        inputs = images.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = compute_losses(inputs, outputs)
        loss = losses[id_loss]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        total_norm = 0
        for p in model.parameters():  # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        epoch_norm = total_norm ** 0.5

        optimizer.step()

        running_mse.append(losses[0].item())
        running_mae.append(losses[1].item())
        # DEBUG

        # if counter == 2:
        #     break
        # counter += 1

    epoch_mse = np.mean(running_mse)
    epoch_mae = np.mean(running_mae)

    return epoch_mse, epoch_mae, epoch_norm


# USED
def val_loop_oe(
        model: AE,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
):
    running_mse = []
    running_mae = []
    model.eval()
    # DEBUG
    # counter = 0
    with torch.no_grad():
        for data in loader:
            images = data
            inputs = images.to(device)
            outputs = model(inputs)
            losses = compute_losses(inputs, outputs)
            running_mse.append(losses[0].item())
            running_mae.append(losses[1].item())
            # DEBUG
            # if counter == 2:
            #     break
            # counter += 1
    epoch_mse = np.mean(running_mse)
    epoch_mae = np.mean(running_mae)
    return epoch_mse, epoch_mae


# USED
def epoch_loop_ae_oe(
        device: torch.device,
        epochs: int,
        model: AE,
        optimizer,
        scheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
):
    # Epoch Loop
    id_loss = 0
    etqdm = tqdm(range(epochs), total=epochs, postfix="Training")
    metrics = defaultdict(list)
    total_epoch_norms = 0
    count_ep = 0
    for _ in etqdm:
        count_ep += 1
        train_loop_returns = train_loop_oe(  # train_loop_returns are [epoch_mse, epoch_mae, epoch_norm]
            model,
            train_loader,
            optimizer,
            device,
            id_loss,
        )
        total_epoch_norms += train_loop_returns[2]
        metrics[f'train_{losses_list[0]}'].append(train_loop_returns[0])
        metrics[f'train_{losses_list[1]}'].append(train_loop_returns[1])

        val_losses = val_loop_oe(model, val_loader, device)  # val_losses are [epoch_mse, epoch_mae]
        scheduler.step(val_losses[id_loss])

        metrics[f'val_{losses_list[0]}'].append(val_losses[0])
        metrics[f'val_{losses_list[1]}'].append(val_losses[1])

        etqdm.set_postfix({f'Train {losses_list[id_loss]}': train_loop_returns[id_loss],
                           f'Val {losses_list[id_loss]}': val_losses[id_loss]})

    return metrics


# USED
def test_loop_oe(device: torch.device, model: AE, test_loader: torch.utils.data.DataLoader):
    model.eval()
    test_losses = {'mse': [], 'mae': []}
    test_labels = []
    # DEBUG
    # count = 0
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), postfix="Running Test Set inference"):
            images = data[0]
            labels = data[1]
            inputs = images.to(device)
            outputs = model(inputs)
            losses = compute_losses(inputs, outputs)
            test_losses["mse"].append(losses[0].item())
            test_losses["mae"].append(losses[1].item())
            test_labels.append(labels[0])
            # DEBUG
            # if count == 25:
            #     return test_losses, test_labels
            # count += 1

    return test_losses, test_labels


# USED
def embedding_production(
        device,
        model,
        train_loader,
        oe_loader,
        test_loader,
        val_loader,
):
    model.eval()
    with torch.no_grad():
        # NORMAL Train_set
        train_set_frame_paths = []
        train_set_embeddings = []
        # debug
        # cts = 0
        for data in tqdm(train_loader, total=len(train_loader), postfix="Producing Train Set embeddings"):
            images = data[0]
            inputs = images.to(device)
            embeddings = model(inputs)
            train_set_embeddings.append(embeddings.cpu())
            train_set_frame_paths.append(data[1][0])
            # debug
            # if cts == 25:
            #     break
            # cts += 1
        train_embs_dict = {
            "embeddings": train_set_embeddings,
            "frame_paths": train_set_frame_paths,
        }
        # Val_set
        val_set_frame_paths = []
        val_set_embeddings = []
        # debug
        # cts = 0
        for data in tqdm(val_loader, total=len(val_loader), postfix="Producing Val Set embeddings"):
            images = data[0]
            inputs = images.to(device)
            embeddings = model(inputs)
            val_set_embeddings.append(embeddings.cpu())
            val_set_frame_paths.append(data[1][0])
            # debug
            # if cts == 25:
            #     break
            # cts += 1

        val_embs_dict = {
            "embeddings": val_set_embeddings,
            "frame_paths": val_set_frame_paths,
        }

        # OE set
        oe_set_frame_paths = []
        oe_set_labels = []
        oe_set_embeddings = []
        # debug
        # cts = 0
        for data in tqdm(oe_loader, total=len(oe_loader),
                         postfix="Producing Outliers embeddings"):
            images = data["image"]
            inputs = images.to(device)
            embeddings = model(inputs)
            oe_set_embeddings.append(embeddings.cpu())
            oe_set_labels.append(data['label'][0].item())
            oe_set_frame_paths.append(data['frame_path'][0])
            # debug
            # if cts == 25:
            #     break
            # cts += 1

        oe_embs_dict = {
            "embeddings": oe_set_embeddings,
            "labels": oe_set_labels,
            # "difficulty": oe_set_difficulty,
            "frame_paths": oe_set_frame_paths,
        }
        # Test_set
        test_set_frame_paths = []
        test_set_labels = []
        test_set_embeddings = []
        # debug
        # cts = 0
        for data in tqdm(test_loader, total=len(test_loader), postfix="Producing Test Set embeddings"):
            images = data["image"]
            inputs = images.to(device)
            embeddings = model(inputs)
            test_set_embeddings.append(embeddings.cpu())
            test_set_labels.append(data['label'][0].item())
            test_set_frame_paths.append(data['frame_path'][0])
            # debug
            # if cts == 25:
            #     break
            # cts += 1

        test_embs_dict = {
            "embeddings": test_set_embeddings,
            "labels": test_set_labels,
            # "difficulty": test_set_difficulty,
            "frame_paths": test_set_frame_paths,
        }
    return train_embs_dict, oe_embs_dict, val_embs_dict, test_embs_dict


# USED
def autoencoder_auc_computation_and_logging(test_labels, test_losses):
    flattened_labels = [0 if el == 0 else 1 for el in test_labels]
    # DEBUG code
    # flattened_labels[0] = 1
    # flattened_labels[1] = 0
    pr_auc = compute_pr_aucs(flattened_labels, losses_list, test_losses)
    roc_auc = compute_roc_aucs(flattened_labels, losses_list, test_losses)
    print(f"Test set {pr_auc=}, {roc_auc=}")
