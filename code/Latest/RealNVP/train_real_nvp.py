import argparse
import logging
from datetime import datetime

import pandas as pd
import torch
from rich.console import Console
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from utils.artifacts_util import oe_rnvp_model_artifact_saver
from utils.check_create_folder import check_create_folder
from utils.rnvp_auc_computation_utils import \
    rnvp_auc_computation_and_logging, per_label_rnvp_metrics
from utils.variables_util import combined_labels_to_names
from models.real_nvp.real_nvp_model_functions import \
    build_network, set_parameters, epoch_loop_oe_normal_rnvp, model_tester
from real_nvp_dataset import OEEmbeddingsDataset, OEEmbeddingsTestSet

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
        mask_type,
        n_layers,
        lr,
        coupling_topology,
        workers,
        max_patience
    ) = set_parameters(
        param
    )
    root_path = param.root_path

    test_set_path = f"{root_path}/data/embeddings/test_embs_dict.pk"
    train_set_path = f"{root_path}/data/embeddings/train_embs_dict.pk"
    validation_set_path = f"{root_path}/data/embeddings/val_embs_dict.pk"
    model_key = f"RNVP_E_{embedding_size}_T_{coupling_topology}_N_{n_layers}_{time_string}"
    save_folder = f"{root_path}/data/rnvp/saves/{model_key}"
    model = build_network(embedding_size, coupling_topology, n_layers, mask_type, batch_norm=False)
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
    val_set = OEEmbeddingsDataset(validation_set_path)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=100,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )
    test_set = OEEmbeddingsTestSet(test_set_path)
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
    metrics, best_model_path = epoch_loop_oe_normal_rnvp(
        device,
        epochs,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        save_folder,
        model_key,
    )

    # TEST
    console.log("Training Completed, Testing Started")

    df_dict = model_tester(best_model_path, model, device, test_loader)

    test_set_df = pd.DataFrame.from_dict(df_dict)
    test_set_df["label"] = pd.to_numeric(test_set_df["label"])

    # Compute AUC fro ml flow logging

    rnvp_auc_computation_and_logging(test_set_df)
    list_of_labels_in_test_set = list(set(test_set.labels))
    # csv row building
    metrics_dict = {}
    for k in list_of_labels_in_test_set:
        v = combined_labels_to_names[k]
        class_metrics_dict = per_label_rnvp_metrics(test_set_df, k)
        if k == 0:
            v = "all"
            metrics_dict[f"{v}_ok_mean_loss"] = class_metrics_dict["ok_mean_loss"]
            metrics_dict[f"{v}_ok_mean_log_prob"] = class_metrics_dict["ok_mean_log_prob"]
            metrics_dict[f"{v}_ok_mean_log_det_J"] = class_metrics_dict["ok_mean_log_det_J"]
            metrics_dict[f"{v}_ok_mean_l2_norm_of_z"] = class_metrics_dict["ok_mean_l2_norm_of_z"]
        metrics_dict[f"{v}_an_mean_loss"] = class_metrics_dict["an_mean_loss"]
        metrics_dict[f"{v}_an_mean_log_prob"] = class_metrics_dict["an_mean_log_prob"]
        metrics_dict[f"{v}_an_mean_log_det_J"] = class_metrics_dict["an_mean_log_det_J"]
        metrics_dict[f"{v}_an_mean_l2_norm_of_z"] = class_metrics_dict["an_mean_l2_norm_of_z"]
        metrics_dict[f"{v}_roc_auc"] = class_metrics_dict["roc_auc"]
        metrics_dict[f"{v}_pr_auc"] = class_metrics_dict["pr_auc"]

    csv_row = {
        **{"model_key": model_key,
           "embedding_size": embedding_size,
           "coupling_topology": coupling_topology,
           "n_layers": n_layers,
           },
        **metrics_dict,
    }
    # SAVE STUFF
    console.log("Testing Completed")
    console.log("Creating and saving Artifacts")
    artifacts_path = save_folder + "/artifacts/"
    oe_rnvp_model_artifact_saver(
        batch_size,
        embedding_size,
        epochs,
        coupling_topology,
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


if __name__ == "__main__":
    main()
