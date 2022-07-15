import argparse
import pickle
from typing import DefaultDict, List, Tuple

import pandas as pd
import torch
from torchinfo import summary

from .check_create_folder import check_create_folder
from ..model.autoencoder import AE


def uniformed_model_artifact_saver(
        batch_size: int,
        bottleneck: int,
        epochs: int,
        layer_1_ft: int,
        lr: float,
        metrics: DefaultDict[str, float],
        test_set_df: pd.DataFrame,
        losses_list: List[str],
        model: AE,
        image_channels: int,
        image_size: Tuple[int, int],
        model_save_folder: str,
        ml_flow_run_id: str,
        id_optimized_loss: int,
        param: argparse.ArgumentParser,
        csv_row: DefaultDict,
        csv_key: str,
        train_sample_batch=None,
        test_sample_batch=None,
):
    check_create_folder(model_save_folder)
    torch.save(model.state_dict(), model_save_folder + f'{csv_key}_last.pth')

    with open(model_save_folder + 'train_info.txt', 'w') as txf:
        if ml_flow_run_id is not None:
            txf.write(f"model type: ae\n"
                      f"lr={lr}\n"
                      f"batch_size={batch_size}\n"
                      f"epochs={epochs}\n"
                      f"bottleneck={bottleneck}\n"
                      f"first_layer_channels={layer_1_ft}\n"
                      f"loss optimized={losses_list[id_optimized_loss]}\n"
                      f"mlflow_run_id = {ml_flow_run_id}")
        else:
            txf.write(f"model type: ae\n"
                      f"lr={lr}\n"
                      f"batch_size={batch_size}\n"
                      f"epochs={epochs}\n"
                      f"bottleneck={bottleneck}\n"
                      f"first_layer_channels={layer_1_ft}\n"
                      f"loss optimized={losses_list[id_optimized_loss]}")

    model.eval()
    summ = summary(model, input_size=(1, image_channels, image_size[0], image_size[1]), device="cpu", depth=4,
                   col_names=["output_size", "kernel_size", "num_params"])

    with open(model_save_folder + 'params.pk', 'wb') as fp:
        pickle.dump(param, fp)

    with open(model_save_folder + 'metrics.pk', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(model_save_folder + 'model_summary.txt', 'w') as txf:
        txf.write(f"{summ}")

    with open(model_save_folder + 'train_sample_batch.pk', "wb") as fp:
        pickle.dump(train_sample_batch, fp)
    with open(model_save_folder + 'test_sample_batch.pk', "wb") as fp:
        pickle.dump(test_sample_batch, fp)
    with open(model_save_folder + f'{csv_key}.pk', "wb") as fp:
        pickle.dump(csv_row, fp)

    test_set_df.to_csv(model_save_folder + f'{csv_key}_test_set_df.csv')
    print(f"artifacts saved at {model_save_folder}")
