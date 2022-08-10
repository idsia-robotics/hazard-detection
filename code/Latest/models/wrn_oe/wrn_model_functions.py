import argparse
import logging
import pickle
import shutil
from collections import defaultdict
from typing import DefaultDict

import mlflow
import numpy as np
import pandas as pd
import torch
import wandb
from torch.nn import functional as F
from torchinfo import summary
from tqdm import tqdm

from utils.check_create_folder import check_create_folder

logging.basicConfig(level=logging.INFO)


def set_wrn_class_parameters(
        device: torch.device,
        ml_flow_run_id: str,
        wandb_run_id: str,
        param: argparse.ArgumentParser,
):
    epochs = param.epochs
    batch_size = param.batch_size
    workers = param.workers
    lr = param.learning_rate
    # max_patience = int(np.log2(epochs)) + 2
    # ratio_oe_samples = param.ratio_oe_samples
    outlier_batch_size = param.oe_batch_size
    analysis_number = param.analysis_number
    subset_size = param.subset_size
    split_date = param.split_date
    momentum = param.momentum
    decay = param.decay
    layers = param.layers
    widen_factor = param.widen_factor
    droprate = param.droprate

    if param.use_ml_flow:
        wandb.config.update(param)
        ml_flow_run_id = mlflow.active_run().info.run_id
        wandb.log({"ml_flow_run_id": ml_flow_run_id})
        mlflow.log_param("wandb_run_id", wandb_run_id)
        mlflow.log_param("analysis_number", analysis_number)
        mlflow.log_param("max_epochs", epochs)
        mlflow.log_param("batch_size", batch_size + outlier_batch_size)
        mlflow.log_param("num_workers", workers)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("decay", decay)
        mlflow.log_param("layers", layers)
        mlflow.log_param("widen_factor", widen_factor)
        mlflow.log_param("droprate", droprate)
        mlflow.log_param("device", device)
        mlflow.log_param("normal_batch_size", batch_size)
        mlflow.log_param("outlier_batch_size", outlier_batch_size)
        mlflow.log_param("split_date", split_date)
        mlflow.log_param("subset_size", subset_size)

    return batch_size, epochs, lr, ml_flow_run_id, workers, momentum, decay, layers, widen_factor, droprate, outlier_batch_size, analysis_number, split_date, subset_size


def train_wrn(model, train_loader_out, train_loader_in, scheduler, optimizer, device):
    model.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without destroying locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    # DEBUG
    # count = 0
    for ok_data, an_data in zip(train_loader_in, train_loader_out):
        ok_input, target = ok_data
        oe_input, _ = an_data
        data = torch.cat((ok_input, oe_input), 0)
        data = data.to(device)
        target = target.to(device)

        # forward
        x = model(data)

        # backward
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(ok_input)], target)
        # cross-entropy from softmax distribution to uniform distribution
        loss += 0.5 * -(x[len(ok_input):].mean(1) - torch.logsumexp(x[len(ok_input):], dim=1)).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        # if count == 25:
        #     return model, loss_avg
        # count += 1
    return model, loss_avg
    # state['train_loss'] = loss_avg


def val_wrn(model, data_loader, device):
    model.eval()
    loss_avg = 0.0
    # DEBUG
    # count = 0
    with torch.no_grad():
        for val_data in data_loader:
            data, target = val_data
            data = data.to(device)
            target = target.to(device)
            # forward
            output = model(data)
            loss = F.cross_entropy(output, target)

            # test loss average
            loss_avg += float(loss.data)
            # DEBUG
            # if count == 25:
            #     return np.mean(loss_avg)
            # count += 1
    return np.mean(loss_avg)


def test_wrn(best_model_path, model, test_loader, device):
    # to_np = lambda x: x.data.cpu().numpy()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    loss_avg = 0.0
    df_dict = {'z': [], 'label': [], 'loss': []}
    # DEBUG
    # count = 0
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), postfix="Running Test Set inference"):
            inputs, label = data
            inputs = inputs.to(device)
            # forward
            output = model(inputs)
            smax = F.softmax(output, dim=1).cpu().numpy()
            ad_score = -np.max(smax, axis=1)
            # binarized_label = 0. if label == 0 else 1.
            # binarized_label = torch.tensor([binarized_label]).unsqueeze(-1).to(device)
            # loss = F.cross_entropy(output, binarized_label)
            # loss = F.binary_cross_entropy_with_logits(output, binarized_label)
            df_dict["z"].append(output.data.cpu())
            df_dict["label"].append(label.item())
            df_dict["loss"].append(ad_score[0])
            # test loss average
            # loss_avg += float(loss.data)
            # DEBUG
            # if count == 25:
            #     return df_dict
            # count += 1

    return df_dict


def best_epoch_saver(epoch, model, param, save_folder, val_loss, model_key):
    best_loss = val_loss
    best_epoch = epoch
    checkpoint_folder = save_folder + f'/checkpoints/'
    check_create_folder(checkpoint_folder)
    best_model_path = checkpoint_folder + f'model_{model_key}_epoch_{epoch}.pth'
    torch.save(model.state_dict(), best_model_path)
    logging.debug(f"Checkpoin model epoch {epoch},saved {best_model_path}")
    if param.use_ml_flow:
        mlflow.log_metric(f'best_epoch', best_epoch)
        wandb.log({f'best_epoch': best_epoch})
    return best_epoch, best_loss, best_model_path


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def oe_wrn_artifact_saver(
        batch_size: int,
        epochs: int,
        lr: float,
        momentum,
        decay,
        layers,
        widen_factor,
        droprate,
        metrics: DefaultDict[str, float],
        test_set_df: pd.DataFrame,
        model,
        artifacts_save_folder: str,
        ml_flow_run_id: str,
        wandb_run_id: str,
        param: argparse.ArgumentParser,
        csv_row: DefaultDict,
        csv_key: str,
        best_model_path: str,
):
    check_create_folder(artifacts_save_folder)
    with open(artifacts_save_folder + 'train_info.txt', 'w') as txf:
        if ml_flow_run_id is not None:
            txf.write(f"model type: ae\n"
                      f"lr={lr}\n"
                      f"momentum={momentum}\n"
                      f"decay={decay}\n"
                      f"layers={layers}\n"
                      f"widen_factor={widen_factor}\n"
                      f"droprate={droprate}\n"
                      f"batch_size={batch_size}\n"
                      f"epochs={epochs}\n"
                      f"wandb_run_id = {wandb_run_id}\n"
                      f"mlflow_run_id = {ml_flow_run_id}\n"
                      f"split_date = {param.split_date}")
        else:
            txf.write(f"model type: ae\n"
                      f"lr={lr}\n"
                      f"momentum={momentum}\n"
                      f"decay={decay}\n"
                      f"layers={layers}\n"
                      f"widen_factor={widen_factor}\n"
                      f"droprate={droprate}\n"
                      f"batch_size={batch_size}\n"
                      f"epochs={epochs}\n"
                      f"mlflow_run_id = {ml_flow_run_id}\n"
                      f"wandb_run_id = {wandb_run_id}\n"
                      f"split_date = {param.split_date}")
    model.eval()
    summ = summary(model, input_size=(1, 3, 64, 64), device="cpu", depth=4,
                   col_names=["input_size", "output_size", "kernel_size", "num_params"])

    with open(artifacts_save_folder + 'params.pk', 'wb') as fp:
        pickle.dump(param, fp)

    with open(artifacts_save_folder + 'metrics.pk', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(artifacts_save_folder + 'model_summary.txt', 'w') as txf:
        txf.write(f"{summ}")

    pd.DataFrame(csv_row, index=[0]).to_csv(artifacts_save_folder + f'{csv_key}.csv')  # in the other code is from dict

    with open(artifacts_save_folder + f'{csv_key}.pk', "wb") as fp:
        pickle.dump(csv_row, fp)

    shutil.copy2(best_model_path, artifacts_save_folder)

    test_set_df.to_csv(artifacts_save_folder + f'{csv_key}_test_set_df.csv')
    print(f"artifacts saved at {artifacts_save_folder}")
