import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.check_create_folder import check_create_folder

logging.basicConfig(level=logging.INFO)


# USED
def set_bin_class_parameters(
        param: argparse.ArgumentParser,
):
    epochs = 500
    batch_size = param.batch_size
    embedding_size = 128
    workers = param.num_workers
    lr = 0.001
    max_patience = int(np.log2(epochs)) + 2
    outlier_batch_size = int(batch_size * 0.1)
    return batch_size, embedding_size, epochs, lr, workers, max_patience, outlier_batch_size


# USED
def epoch_loop_bin_class(device: torch.device,
                         epochs: int,
                         model,
                         optimizer,
                         scheduler,
                         param: argparse.ArgumentParser,
                         normal_train_loader: torch.utils.data.DataLoader,
                         anomalies_train_loader: torch.utils.data.DataLoader,
                         val_loader: torch.utils.data.DataLoader,
                         save_folder: str,
                         model_key: str,
                         ):
    # Epoch Loop
    logging.info('Training started')

    etqdm = tqdm(range(epochs), total=epochs, postfix="Training")
    best_loss = None
    best_epoch = 0
    best_model_path = None
    metrics = defaultdict(list)
    for epoch in etqdm:
        model, train_loss = trainer_bin_class(
            device,
            model,
            optimizer,
            normal_train_loader,
            anomalies_train_loader,
        )
        metrics[f'train_loss'].append(train_loss)
        val_loss = validator(val_loader, model, device)
        scheduler.step(val_loss)
        metrics[f'val_loss'].append(val_loss)
        if best_loss is None:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model, param,
                                                                      save_folder, val_loss, model_key)
        if val_loss < best_loss:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model, param,
                                                                      save_folder, val_loss, model_key)

        etqdm.set_description(
            f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | best model @ epoch {best_epoch}")
    logging.info('Finished training.')
    return metrics, best_model_path


# USED
def best_epoch_saver(epoch, model, param, save_folder, val_loss, model_key):
    best_loss = val_loss
    best_epoch = epoch
    checkpoint_folder = save_folder + f'/checkpoints/'
    check_create_folder(checkpoint_folder)
    best_model_path = checkpoint_folder + f'model_{model_key}_epoch_{epoch}.pth'
    torch.save(model.state_dict(), best_model_path)
    logging.debug(f"Checkpoin model epoch {epoch},saved {best_model_path}")
    return best_epoch, best_loss, best_model_path


# USED
def model_tester(best_model_path, model, device, test_loader):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    df_dict = {'z': [], 'label': [], 'loss': []}
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), postfix="Running Test Set inference"):
            inputs, label, _ = data
            inputs = inputs.to(device)
            label = label.unsqueeze(-1).float().to(device)
            outputs = model(inputs)
            loss = F.binary_cross_entropy(outputs, label)
            df_dict["z"].append(outputs.item())
            df_dict["label"].append(label.item())
            df_dict["loss"].append(loss.detach().cpu().item())
    return df_dict


# USED
def validator(val_loader, model, device):
    with torch.no_grad():
        model.eval()
        losses = []
        for data in val_loader:
            inputs = data[0]
            inputs = inputs.to(device)
            target = torch.zeros(inputs.shape[0]).unsqueeze(-1).to(device)
            outputs = model(inputs)
            loss = F.binary_cross_entropy(outputs, target)
            losses.append(loss.detach().cpu().item())
    return np.mean(losses)


# USED
def trainer_bin_class(device, model, optimizer, normal_train_loader, anomalies_train_loader):
    model.train()
    losses = []
    for ok_data, an_data in zip(normal_train_loader, anomalies_train_loader):
        # normal data
        ok_inputs, _ = ok_data
        an_inputs, _, _ = an_data
        inputs = torch.cat((ok_inputs, an_inputs))
        ok_t = torch.zeros(ok_inputs.shape[0])
        an_t = torch.ones(an_inputs.shape[0])
        target = torch.cat((ok_t, an_t)).unsqueeze(-1).to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.binary_cross_entropy(outputs, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, np.mean(losses)
