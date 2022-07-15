import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
from torch import linalg as LA
from tqdm import tqdm

from utils.check_create_folder import check_create_folder
from models.real_nvp import real_nvp

logging.basicConfig(level=logging.INFO)


# USED
def build_network(embedding_size, coupling_topology, n_layers, mask_type, batch_norm=True):
    """Builds the neural network."""
    model = real_nvp.LinearRNVP(input_dim=embedding_size,
                                coupling_topology=coupling_topology,
                                flow_n=n_layers,
                                batch_norm=batch_norm,
                                mask_type=mask_type,
                                conditioning_size=None,
                                use_permutation=False,
                                single_function=False)

    return model


# USED
def epoch_loop_oe_normal_rnvp(device: torch.device,
                              epochs: int,
                              model: real_nvp,
                              optimizer,
                              scheduler,
                              train_loader: torch.utils.data.DataLoader,
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
        model, train_loss, train_log_prob, train_log_det_j = trainer(
            device, model, optimizer, train_loader
        )

        metrics[f'train_loss'].append(train_loss)
        val_loss, val_log_prob, val_log_det_j = validator(val_loader, model, device)
        scheduler.step(val_loss)
        metrics[f'val_loss'].append(val_loss)
        if best_loss is None:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model,
                                                                      save_folder, val_loss, model_key)
        if val_loss < best_loss:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model,
                                                                      save_folder, val_loss, model_key)

        etqdm.set_description(
            f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | best model @ epoch {best_epoch}")
    logging.info('Finished training.')

    return metrics, best_model_path


# USED
def epoch_loop_oe_with_oe_rnvp(device: torch.device,
                               epochs: int,
                               model: real_nvp,
                               optimizer,
                               scheduler,
                               normal_train_loader: torch.utils.data.DataLoader,
                               anomalies_train_loader: torch.utils.data.DataLoader,
                               val_loader: torch.utils.data.DataLoader,
                               save_folder: str,
                               etn_key: str,
                               gamma: float,
                               lambda_p: float,
                               ):
    # Epoch Loop
    logging.info('Training started')

    etqdm = tqdm(range(epochs), total=epochs, postfix="Training")
    best_loss = None
    best_epoch = 0
    best_model_path = None
    metrics = defaultdict(list)
    for epoch in etqdm:
        model, train_loss, train_log_prob, train_log_det_j, oe_loss_component_list, ok_l2_mean, an_l2_mean = trainer_oe(
            device,
            model,
            optimizer,
            normal_train_loader,
            anomalies_train_loader,
            gamma=gamma,
            lambda_p=lambda_p
        )
        metrics[f'train_loss'].append(train_loss)

        val_loss, val_log_prob, val_log_det_j = validator(val_loader, model, device)
        scheduler.step(val_loss)

        metrics[f'val_loss'].append(val_loss)
        if best_loss is None:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model,
                                                                      save_folder, val_loss, etn_key)
        if val_loss < best_loss:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model,
                                                                      save_folder, val_loss, etn_key)

        etqdm.set_description(
            f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | best model @ epoch {best_epoch}")
    logging.info('Finished training.')

    return metrics, best_model_path


# USED
def best_epoch_saver(epoch, model, save_folder, val_loss, etn_key):
    best_loss = val_loss
    best_epoch = epoch
    checkpoint_folder = save_folder + f'/checkpoints/'
    check_create_folder(checkpoint_folder)
    best_model_path = checkpoint_folder + f'model_{etn_key}_epoch_{epoch}.pth'
    torch.save(model.state_dict(), best_model_path)
    logging.debug(f"Checkpoin model epoch {epoch},saved {best_model_path}")
    return best_epoch, best_loss, best_model_path


# USED
def set_parameters(
        param: argparse.ArgumentParser,
):
    epochs = 500
    batch_size = param.batch_size
    embedding_size = 128
    workers = param.num_workers
    lr = 0.001
    max_patience = int(np.log2(epochs)) + 2
    coupling_topology = [128]
    num_layers = 4
    mask_type = 'odds'

    return batch_size, embedding_size, epochs, mask_type, num_layers, lr, coupling_topology, workers, max_patience


# USED
def set_parameters_rnvp_inference(
        param: argparse.ArgumentParser,
):
    embedding_size = 128
    workers = param.num_workers
    coupling_topology = [128]
    num_layers = 4
    mask_type = 'odds'
    layer_1_ft = param.first_layer_size
    layer_2_ft = layer_1_ft * 2
    layer_3_ft = layer_1_ft * 2 * 2
    layer_4_ft = layer_1_ft * 2 * 2 * 2
    input_channel = 3
    image_size = (64, 64)
    widths_ = [
        input_channel,
        layer_1_ft,
        layer_2_ft,
        layer_3_ft,
        layer_4_ft,
    ]
    return embedding_size, mask_type, num_layers, coupling_topology, workers, widths_, image_size


# USED
def set_oe_parameters(
        param: argparse.ArgumentParser,
):
    epochs = 500
    batch_size = param.batch_size
    embedding_size = 128
    workers = param.num_workers
    lr = 0.001
    max_patience = int(np.log2(epochs)) + 2
    coupling_topology = [128]
    num_layers = 4
    mask_type = 'odds'
    outlier_batch_size = int(batch_size * 0.1)
    gamma = 100
    lambda_p = 1

    return batch_size, embedding_size, epochs, mask_type, num_layers, lr, coupling_topology, workers, max_patience, gamma, lambda_p, outlier_batch_size


# USED
def model_tester(best_model_path, model, device, test_loader):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    df_dict = {'z': [], 'label': [], 'loss': [], 'log_prob': [], 'log_det_J': [], 'l2_norm_of_z': []}
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), postfix="Running Test Set inference"):
            inputs, label, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            z, log_det_J = outputs
            log_prob_z = model.log_prob(z)
            log_prob_z_mean = log_prob_z.mean()
            log_det_j_mean = log_det_J.mean()
            loss = - log_prob_z_mean - log_det_j_mean
            logging.debug(f"TEST: inputs.shape={inputs.shape}\n"
                          f"      outputs[0].shape={outputs[0].shape}\n"
                          f"      loss {loss} = - {log_prob_z_mean} - {log_det_j_mean}")
            df_dict["z"].append(z.detach().squeeze(0).cpu().numpy())
            df_dict["label"].append(label.item())
            df_dict["loss"].append(loss.detach().cpu().item())
            df_dict["log_prob"].append(log_prob_z.detach().cpu().item())
            df_dict["log_det_J"].append(log_det_J.detach().cpu().item())
            df_dict["l2_norm_of_z"].append(LA.norm(z).detach().cpu().item())
    return df_dict


# USED
def validator(val_loader, model, device):
    with torch.no_grad():
        model.eval()
        losses = []
        log_prob_mean_list = []
        log_det_j_list = []
        for data in val_loader:
            inputs = data[0]
            inputs = inputs.to(device)
            outputs = model(inputs)
            z, log_det_J = outputs
            log_prob_z = model.log_prob(z)
            log_prob_z_mean = log_prob_z.mean()
            log_det_j_mean = log_det_J.mean()
            log_prob_mean_list.append(log_prob_z_mean.item())
            log_det_j_list.append(log_det_j_mean.item())
            loss = - log_prob_z_mean - log_det_j_mean
            logging.debug(f"VAL: inputs.shape={inputs.shape}\n"
                          f"     outputs[0].shape={outputs[0].shape}\n"
                          f"     loss {loss} = - {log_prob_z_mean} - {log_det_j_mean}")
            losses.append(loss.detach().cpu().item())
    return np.mean(losses), np.mean(log_prob_mean_list), np.mean(log_det_j_list)


# USED
def trainer_oe(device, model, optimizer, normal_train_loader, anomalies_train_loader, gamma, lambda_p):
    model.train()
    losses = []
    log_prob_z_mean_list = []
    log_det_j_mean_list = []
    oe_loss_component_list = []
    ok_l2norm_list = []
    an_l2norm_list = []
    for ok_data, an_data in zip(normal_train_loader, anomalies_train_loader):
        # normal data
        ok_inputs, _ = ok_data
        ok_inputs = ok_inputs.to(device)
        optimizer.zero_grad()
        ok_outputs = model(ok_inputs)
        ok_z, ok_log_det_J = ok_outputs
        ok_log_prob_z = model.log_prob(ok_z)
        ok_log_prob_z_mean = ok_log_prob_z.mean()
        ok_log_det_j_mean = ok_log_det_J.mean()

        an_inputs, _, _ = an_data
        an_inputs = an_inputs.to(device)

        model.eval()
        an_outputs = model(an_inputs)
        model.train()

        an_z, an_log_det_J = an_outputs
        an_log_prob_z = model.log_prob(an_z)
        maxs = torch.maximum(torch.zeros((len(an_log_prob_z)), device=device),
                             gamma
                             +
                             (- ok_log_prob_z[:len(an_log_prob_z)] - ok_log_det_J[:len(an_log_det_J)])
                             -
                             (- an_log_prob_z - an_log_det_J))
        oe_loss_component = maxs.mean()
        log_prob_z_mean_list.append(ok_log_prob_z_mean.item())
        log_det_j_mean_list.append(ok_log_det_j_mean.item())
        oe_loss_component_list.append(oe_loss_component.item())
        loss = - ok_log_prob_z_mean - ok_log_det_j_mean + lambda_p * oe_loss_component
        logging.debug(f"TRAIN: ok_inputs.shape={ok_inputs.shape}\n"
                      f"       ok_outputs[0].shape={ok_outputs[0].shape}\n"
                      f"       ok_log_prob_z = {ok_log_prob_z}\n"
                      f"       an_log_prob_z = {an_log_prob_z}\n"
                      f"       maxs = {maxs}\n"
                      f"       loss {loss} = - {ok_log_prob_z_mean} - {ok_log_det_j_mean} + {lambda_p} * {oe_loss_component}\n")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        ok_l2norm_list += LA.norm(ok_z, dim=1).detach().cpu().tolist()
        an_l2norm_list += LA.norm(an_z, dim=1).detach().cpu().tolist()
    return model, np.mean(losses), np.mean(log_prob_z_mean_list), np.mean(log_det_j_mean_list), np.mean(
        oe_loss_component_list), np.mean(ok_l2norm_list), np.mean(an_l2norm_list)


# USED
def trainer(device, model, optimizer, train_loader):
    model.train()
    losses = []
    log_prob_z_mean_list = []
    log_det_j_mean_list = []
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        z, log_det_J = outputs
        log_prob_z = model.log_prob(z)
        log_prob_z_mean = log_prob_z.mean()
        log_det_j_mean = log_det_J.mean()
        log_prob_z_mean_list.append(log_prob_z_mean.item())
        log_det_j_mean_list.append(log_det_j_mean.item())
        loss = - log_prob_z_mean - log_det_j_mean
        logging.debug(f"TRAIN: inputs.shape={inputs.shape}\n"
                      f"       outputs[0].shape={outputs[0].shape}\n"
                      f"       loss {loss} = - {log_prob_z_mean} - {log_det_j_mean}")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, np.mean(losses), np.mean(log_prob_z_mean_list), np.mean(log_det_j_mean_list)
