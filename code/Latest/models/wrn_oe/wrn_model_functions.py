import argparse
import logging
import pickle
import shutil
from typing import DefaultDict

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchinfo import summary
from tqdm import tqdm

from utils.check_create_folder import check_create_folder

logging.basicConfig(level=logging.INFO)


def set_wrn_class_parameters(
        param: argparse.ArgumentParser,
):
    epochs = param.epochs
    batch_size = param.batch_size
    workers = param.workers
    lr = param.learning_rate
    outlier_batch_size = param.oe_batch_size
    analysis_number = param.analysis_number
    subset_size = param.subset_size
    split_date = param.split_date
    momentum = param.momentum
    decay = param.decay
    layers = param.layers
    widen_factor = param.widen_factor
    droprate = param.droprate
    return batch_size, epochs, lr, workers, momentum, decay, layers, widen_factor, droprate, outlier_batch_size, analysis_number, split_date, subset_size


def train_wrn(model, train_loader_out, train_loader_in, scheduler, optimizer, device):
    model.train()  # enter train mode
    loss_avg = 0.0
    # start at a random point of the outlier dataset; this induces more randomness without destroying locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
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
    return model, loss_avg


def val_wrn(model, data_loader, device):
    model.eval()
    loss_avg = 0.0
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
    return np.mean(loss_avg)


def test_wrn(best_model_path, model, test_loader, device):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    loss_avg = 0.0
    df_dict = {'z': [], 'label': [], 'loss': []}
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), postfix="Running Test Set inference"):
            inputs, label = data
            inputs = inputs.to(device)
            # forward
            output = model(inputs)
            smax = F.softmax(output, dim=1).cpu().numpy()
            ad_score = -np.max(smax, axis=1)
            df_dict["z"].append(output.data.cpu())
            df_dict["label"].append(label.item())
            df_dict["loss"].append(ad_score[0])
    return df_dict


def best_epoch_saver(epoch, model, param, save_folder, val_loss, model_key):
    best_loss = val_loss
    best_epoch = epoch
    checkpoint_folder = save_folder + f'/checkpoints/'
    check_create_folder(checkpoint_folder)
    best_model_path = checkpoint_folder + f'model_{model_key}_epoch_{epoch}.pth'
    torch.save(model.state_dict(), best_model_path)
    logging.debug(f"Checkpoin model epoch {epoch},saved {best_model_path}")
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
        param: argparse.ArgumentParser,
        csv_row: DefaultDict,
        csv_key: str,
        best_model_path: str,
):
    check_create_folder(artifacts_save_folder)
    with open(artifacts_save_folder + 'train_info.txt', 'w') as txf:
        txf.write(f"model type: ae\n"
                  f"lr={lr}\n"
                  f"momentum={momentum}\n"
                  f"decay={decay}\n"
                  f"layers={layers}\n"
                  f"widen_factor={widen_factor}\n"
                  f"droprate={droprate}\n"
                  f"batch_size={batch_size}\n"
                  f"epochs={epochs}\n"
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
