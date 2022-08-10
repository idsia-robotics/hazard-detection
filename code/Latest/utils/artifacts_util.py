import argparse
import pickle
import shutil
from typing import DefaultDict, List, Tuple

import pandas as pd
import torch
from torchinfo import summary

from check_create_folder import check_create_folder
from models.autoencoder.autoencoder import AE



def oe_bin_class_model_artifact_saver(
        batch_size: int,
        embedding_size: int,
        epochs: int,
        lr: float,
        metrics: DefaultDict[str, float],
        test_set_df: pd.DataFrame,
        model: AE,
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
                      f"batch_size={batch_size}\n"
                      f"epochs={epochs}\n"
                      f"embedding_size={embedding_size}\n"
                      f"split_date = {param.split_date}")
    model.eval()
    summ = summary(model, input_size=(1, embedding_size), device="cpu", depth=4,
                   col_names=["input_size", "output_size", "kernel_size", "num_params"])

    with open(artifacts_save_folder + 'params.pk', 'wb') as fp:
        pickle.dump(param, fp)

    with open(artifacts_save_folder + 'metrics.pk', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(artifacts_save_folder + 'model_summary.txt', 'w') as txf:
        txf.write(f"{summ}")

    pd.DataFrame(csv_row, index=[0]).to_csv(artifacts_save_folder + f'{csv_key}.csv') # in the other code is from dict

    with open(artifacts_save_folder + f'{csv_key}.pk', "wb") as fp:
        pickle.dump(csv_row, fp)

    shutil.copy2(best_model_path, artifacts_save_folder)
    test_set_df.to_csv(artifacts_save_folder + f'{csv_key}_test_set_df.csv')
    print(f"artifacts saved at {artifacts_save_folder}")


def oe_rnvp_model_artifact_saver(
        batch_size: int,
        embedding_size: int,
        epochs: int,
        coupling_topology: List[int],
        lr: float,
        metrics: DefaultDict[str, float],
        test_set_df: pd.DataFrame,
        model: AE,
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
                  f"batch_size={batch_size}\n"
                  f"epochs={epochs}\n"
                  f"embedding_size={embedding_size}\n"
                  f"coupling_topology={coupling_topology}\n"
                  f"split_date = {param.split_date}")

    model.eval()
    summ = summary(model, input_size=(1, embedding_size), device="cpu", depth=4,
                   col_names=["input_size", "output_size", "kernel_size", "num_params"])

    with open(artifacts_save_folder + 'params.pk', 'wb') as fp:
        pickle.dump(param, fp)

    with open(artifacts_save_folder + 'metrics.pk', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(artifacts_save_folder + 'model_summary.txt', 'w') as txf:
        txf.write(f"{summ}")

    pd.DataFrame.from_dict(csv_row).to_csv(artifacts_save_folder + f'{csv_key}.csv')

    with open(artifacts_save_folder + f'{csv_key}.pk', "wb") as fp:
        pickle.dump(csv_row, fp)

    shutil.copy2(best_model_path, artifacts_save_folder)
    test_set_df.to_csv(artifacts_save_folder + f'{csv_key}_test_set_df.csv')
    test_set_df.to_pickle(artifacts_save_folder + f'{csv_key}_test_set_df.pk')
    print(f"artifacts saved at {artifacts_save_folder}")


# USED
def small_ae_pretrain_model_artifact_saver(
        artifacts_save_folder: str,
        train_embs_dict,
        oe_embs_dict,
        val_embs_dict,
        test_embs_dict,

):
    check_create_folder(artifacts_save_folder)
    with open(artifacts_save_folder + f'train_embs_dict.pk', "wb") as fp:
        pickle.dump(train_embs_dict, fp)
    with open(artifacts_save_folder + f'oe_embs_dict.pk', "wb") as fp:
        pickle.dump(oe_embs_dict, fp)
    with open(artifacts_save_folder + f'val_embs_dict.pk', "wb") as fp:
        pickle.dump(val_embs_dict, fp)
    with open(artifacts_save_folder + f'test_embs_dict.pk', "wb") as fp:
        pickle.dump(test_embs_dict, fp)

    print(f"artifacts saved at {artifacts_save_folder}")


# USED
def ae_pretrain_model_artifact_saver(
        batch_size: int,
        bottleneck: int,
        epochs: int,
        layer_1_ft: int,
        lr: float,
        metrics: DefaultDict[str, float],
        model: AE,
        image_channels: int,
        image_size: Tuple[int, int],
        artifacts_save_folder: str,
        param: argparse.ArgumentParser,
        bfs_key: str,
        train_sample_batch=None,
        test_sample_batch=None,
):
    check_create_folder(artifacts_save_folder)
    torch.save(model.state_dict(), artifacts_save_folder + f'{bfs_key}_last.pth')

    with open(artifacts_save_folder + 'train_info.txt', 'w') as txf:
        txf.write(f"model type: ae\n"
                  f"lr={lr}\n"
                  f"batch_size={batch_size}\n"
                  f"epochs={epochs}\n"
                  f"bottleneck={bottleneck}\n"
                  f"first_layer_channels={layer_1_ft}\n"
                  f"data_channels={param.input_channels}")

    model.eval()
    summ = summary(model, input_size=(1, image_channels, image_size[0], image_size[1]), device="cpu", depth=4,
                   col_names=["output_size", "kernel_size", "num_params"])

    with open(artifacts_save_folder + 'params.pk', 'wb') as fp:
        pickle.dump(param, fp)

    with open(artifacts_save_folder + 'metrics.pk', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(artifacts_save_folder + 'model_summary.txt', 'w') as txf:
        txf.write(f"{summ}")

    with open(artifacts_save_folder + 'train_sample_batch.pk', "wb") as fp:
        pickle.dump(train_sample_batch, fp)
    with open(artifacts_save_folder + 'test_sample_batch.pk', "wb") as fp:
        pickle.dump(test_sample_batch, fp)

    print(f"artifacts saved at {artifacts_save_folder}")


# USED
def ae_embedding_saver(
        batch_size: int,
        bottleneck: int,
        epochs: int,
        layer_1_ft: int,
        lr: float,
        metrics: DefaultDict[str, float],
        losses_list: List[str],
        model: AE,
        image_channels: int,
        image_size: Tuple[int, int],
        artifacts_save_folder: str,
        ml_flow_run_id: str,
        id_optimized_loss: int,
        param: argparse.ArgumentParser,
        train_embs_dict,
        oe_embs_dict,
        val_embs_dict,
        test_embs_dict,
        bfs_key: str,
        train_sample_batch=None,
        test_sample_batch=None,
):
    check_create_folder(artifacts_save_folder)
    torch.save(model.state_dict(), artifacts_save_folder + f'{bfs_key}_last.pth')

    with open(artifacts_save_folder + 'train_info.txt', 'w') as txf:
        if ml_flow_run_id is not None:
            txf.write(f"model type: ae\n"
                      f"lr={lr}\n"
                      f"batch_size={batch_size}\n"
                      f"epochs={epochs}\n"
                      f"bottleneck={bottleneck}\n"
                      f"first_layer_channels={layer_1_ft}\n"
                      f"loss optimized={losses_list[id_optimized_loss]}\n"
                      f"mlflow_run_id={ml_flow_run_id}\n"
                      f"data_channels={param.input_channels}")
        else:
            txf.write(f"model type: ae\n"
                      f"lr={lr}\n"
                      f"batch_size={batch_size}\n"
                      f"epochs={epochs}\n"
                      f"bottleneck={bottleneck}\n"
                      f"first_layer_channels={layer_1_ft}\n"
                      f"loss optimized={losses_list[id_optimized_loss]}\n"
                      f"data_channels={param.input_channels}")

    model.eval()
    summ = summary(model, input_size=(1, image_channels, image_size[0], image_size[1]), device="cpu", depth=4,
                   col_names=["output_size", "kernel_size", "num_params"])

    with open(artifacts_save_folder + 'params.pk', 'wb') as fp:
        pickle.dump(param, fp)

    with open(artifacts_save_folder + 'metrics.pk', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(artifacts_save_folder + 'model_summary.txt', 'w') as txf:
        txf.write(f"{summ}")

    with open(artifacts_save_folder + 'train_sample_batch.pk', "wb") as fp:
        pickle.dump(train_sample_batch, fp)
    with open(artifacts_save_folder + 'test_sample_batch.pk', "wb") as fp:
        pickle.dump(test_sample_batch, fp)

    with open(artifacts_save_folder + f'{bfs_key}_train_embs_dict.pk', "wb") as fp:
        pickle.dump(train_embs_dict, fp)
    with open(artifacts_save_folder + f'{bfs_key}_oe_embs_dict.pk', "wb") as fp:
        pickle.dump(oe_embs_dict, fp)
    with open(artifacts_save_folder + f'{bfs_key}_val_embs_dict.pk', "wb") as fp:
        pickle.dump(val_embs_dict, fp)
    with open(artifacts_save_folder + f'{bfs_key}_test_embs_dict.pk', "wb") as fp:
        pickle.dump(test_embs_dict, fp)

    print(f"artifacts saved at {artifacts_save_folder}")
