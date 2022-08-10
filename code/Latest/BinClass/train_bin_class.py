import argparse
import logging
from datetime import datetime
from typing import List

import pandas as pd
import torch
from rich.console import Console
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from models.binary_classifier.bin_class import BinaryClassifier
from models.binary_classifier.bin_class_model_functions import set_bin_class_parameters, epoch_loop_bin_class, \
    model_tester
from step_7_outlier_exposure.oe_rnvp_dataset import OEEmbeddingsDataset, OEEmbeddingsTestSet, OEEmbeddingsOutliersSet
from utils.artifacts_util import oe_bin_class_model_artifact_saver
from utils.check_create_folder import check_create_folder
from utils.metrics_util import compute_pr_aucs_single_loss, compute_roc_aucs_single_loss
from utils.variables_util import combined_labels_to_names

console = Console()
logging.basicConfig(level=logging.INFO)


def params_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        "-b", type=int, default=2048)
    parser.add_argument("--num_workers",
                        '-w', type=int, default=4)
    parser.add_argument('--gpu_number',
                        '-g', type=int, default=0)
    parser.add_argument('--root_path',
                        '-r', type=str, default=".")
    param = parser.parse_args()
    return param


def main():
    """
    if param.gpu_number = -1 it uses the cpu
    Returns:
    """
    # MODEL INIT

    param = params_parser()

    console.log(f'Using the following params:{param}')
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if param.gpu_number == -1:
        cuda_gpu = f"cpu"
    else:
        cuda_gpu = f"cuda:{param.gpu_number}"
    device = torch.device(cuda_gpu if torch.cuda.is_available() else "cpu")
    print(device)

    (
        batch_size,
        embedding_size,
        epochs,
        lr,
        workers,
        max_patience,
        outlier_batch_size,
    ) = set_bin_class_parameters(
        param
    )
    root_path = param.root_path

    test_set_path = f"{root_path}/data/embeddings/test_embs_dict.pk"
    train_set_path = f"{root_path}/data/embeddings/train_embs_dict.pk"
    validation_set_path = f"{root_path}/data/embeddings/val_embs_dict.pk"
    outliers_set_path = f"{root_path}/data/embeddings/oe_embs_dict.pk"
    model_key = f"binary_classifier_{time_string}"
    save_folder = f"{root_path}/data/bin_class/saves/{model_key}"
    model = BinaryClassifier()
    model.to(device)
    check_create_folder(save_folder)

    # DATA INIT
    train_set = OEEmbeddingsDataset(train_set_path)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )
    outliers_set_required_size = outlier_batch_size * len(train_loader)
    outliers_set = OEEmbeddingsOutliersSet(
        file_path=outliers_set_path,
        required_dataset_size=outliers_set_required_size
    )

    outliers_loader = torch.utils.data.DataLoader(
        outliers_set,
        batch_size=outlier_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_set = OEEmbeddingsDataset(validation_set_path)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=100,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )
    test_set = OEEmbeddingsTestSet(
        test_set_path,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=max_patience, verbose=True)

    # TRAIN
    metrics, best_model_path = epoch_loop_bin_class(
        device,
        epochs,
        model,
        optimizer,
        scheduler,
        param,
        train_loader,
        outliers_loader,
        val_loader,
        save_folder,
        model_key,
    )

    # TEST
    console.log("Training Completed, Testing Started")

    df_dict = model_tester(best_model_path, model, device, test_loader)

    # df_dict = test_loop_df_rows(device, model, test_loader)
    test_set_df = pd.DataFrame.from_dict(df_dict)
    test_set_df["label"] = pd.to_numeric(test_set_df["label"])

    # Compute AUC for ml flow logging

    rnvp_auc_computation_and_logging(test_set_df, param)
    list_of_labels_in_test_set = list(set(test_set.labels))

    # csv row building
    metrics_dict = {}
    for k in list_of_labels_in_test_set:
        v = combined_labels_to_names[k]
        class_metrics_dict = per_label_rnvp_metrics(test_set_df, k)
        if k == 0:
            v = "all"
            metrics_dict[f"{v}_ok_mean_loss"] = class_metrics_dict["ok_mean_loss"]
        metrics_dict[f"{v}_an_mean_loss"] = class_metrics_dict["an_mean_loss"]
        metrics_dict[f"{v}_roc_auc"] = class_metrics_dict["roc_auc"]
        metrics_dict[f"{v}_pr_auc"] = class_metrics_dict["pr_auc"]

    csv_row = {
        **{"model_key": model_key,
           "embedding_size": embedding_size,
           },
        **metrics_dict,
    }

    # SAVE STUFF
    console.log("Testing Completed")
    console.log("Creating and saving Artifacts")
    artifacts_path = save_folder + "/artifacts/"
    oe_bin_class_model_artifact_saver(
        batch_size,
        embedding_size,
        epochs,
        lr,
        metrics,
        test_set_df,
        model,
        artifacts_path,
        param=param,
        csv_row=csv_row,
        csv_key=model_key,
        best_model_path=best_model_path,
    )

    console.log(f"Script completed, artifacts located at {save_folder}.")


def avg_l2_norms(test_df):
    ok_l2 = test_df.loc[test_df.label == 0].l2_norm_of_z.values.mean()
    an_l2 = test_df.loc[test_df.label != 0].l2_norm_of_z.values.mean()
    return an_l2, ok_l2


def rnvp_auc_computation_and_logging(test_df, param):
    labels = [0 if el == 0 else 1 for el in test_df["label"].values]
    metrics_dict = compute_oe_rnvp_model_metrics(
        labels,
        test_df
    )
    print(f'test_set_ok_mean_loss = {metrics_dict["ok_mean_loss"]}\n'
              f'test_set_an_mean_loss = {metrics_dict["an_mean_loss"]}\n'
              f'test_set_roc_auc = {metrics_dict["roc_auc"]}\n'
              f'test_set_pr_auc = {metrics_dict["pr_auc"]}\n'
              )


def per_label_rnvp_metrics(df, label_key):
    if label_key == 0:
        label_unique_values = [0 if el == 0 else 1 for el in df["label"].values]
        return_dict = compute_oe_rnvp_model_metrics(
            label_unique_values,
            df,
        )
    else:

        df_anomaly = df[df.label.isin([0, label_key])]
        label_unique_values = [0 if el == 0 else 1 for el in df_anomaly["label"].values]
        return_dict = compute_oe_rnvp_model_metrics(
            label_unique_values,
            df_anomaly,
        )
    return return_dict


def compute_oe_rnvp_model_metrics(
        labels: List[int],
        df_losses: pd.DataFrame,
):
    y_true = labels
    losses = df_losses["z"].values
    pr_auc = compute_pr_aucs_single_loss(y_true, losses)
    roc_auc = compute_roc_aucs_single_loss(y_true, losses)
    an_mean_loss = df_losses[df_losses["label"] != 0]["loss"].values.mean()
    ok_mean_loss = df_losses[df_losses["label"] == 0]["loss"].values.mean()
    # composing return dict
    return_dict = {
        "an_mean_loss": an_mean_loss,
        "ok_mean_loss": ok_mean_loss,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    return return_dict


if __name__ == "__main__":
    main()
