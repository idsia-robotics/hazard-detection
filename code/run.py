import argparse
from datetime import datetime

import albumentations as A

import mlflow
import pandas as pd
import torch
from rich.console import Console
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from paper_code_release.model.autoencoder import AE
from paper_code_release.model.uniformed_model_functions import set_model_and_train_parameters_patches, epoch_loop_patches, \
    test_loop_df_rows
from paper_code_release.paper_utils.artifacts_util import uniformed_model_artifact_saver
from paper_code_release.dataset import UniformedPatchesDataset, UniformedPatchesTestset

from paper_code_release.paper_utils.check_create_folder import check_create_folder
from paper_code_release.paper_utils.init_utils import uniformed_model_paths_init
from paper_code_release.paper_utils.losses_util import losses_list
from paper_code_release.paper_utils.metrics_util import compute_uniformed_model_metrics
from paper_code_release.paper_utils.variables_util import dataset_names, available_scale_levels, datasets_labels_names, scaled_image_shapes

console = Console()


def params_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs",
                        '-e', type=int, default=15)
    parser.add_argument("--batch_size",
                        "-b", type=int, default=320)
    parser.add_argument("--bottleneck",
                        "-n", type=int, default=128)
    parser.add_argument("--num_workers",
                        '-w', type=int, default=4)
    parser.add_argument('--gpu_number',
                        '-g', type=int, default=0)
    parser.add_argument('--learning_rate',
                        '-l', type=float, default=1e-3)
    parser.add_argument('--use_ml_flow',
                        '-m', type=int, default=1)
    parser.add_argument('--first_layer_size',
                        '-f', type=int, default=128)
    parser.add_argument('--input_channels',
                        '-i', type=int, default=3)
    parser.add_argument('--id_optimized_loss',
                        '-o', type=int, default=0)
    parser.add_argument('--render_video',
                        '-v', type=int, default=0)
    parser.add_argument('--patience_thres',
                        '-p', type=float, default=0.001)
    parser.add_argument('--dataset',
                        '-d', type=int, default=1)
    parser.add_argument('--scale_level',
                        '-s', type=int, default=2)
    parser.add_argument('--test_patches_number',
                        '-t', type=int, default=250)
    param = parser.parse_args()
    return param


def per_label_metrics(df, label_key):
    if label_key == 0:
        label_unique_values = [0 if el[0] == 0 else 1 for el in
                               df[["frame_id", "frame_label"]].groupby("frame_id")["frame_label"].unique().values]
        return_dict = compute_uniformed_model_metrics(
            label_unique_values,
            losses_list,
            df[["frame_id", "mse_loss", "mae_loss"]],
            stdev=True,
        )
    else:
        df_anomaly = df[df["frame_label"].isin([0, label_key])]
        label_unique_values = [0 if el[0] == 0 else 1 for el in
                               df_anomaly[["frame_id", "frame_label"]].groupby("frame_id")[
                                   "frame_label"].unique().values]

        return_dict = compute_uniformed_model_metrics(
            label_unique_values,
            losses_list,
            df_anomaly[["frame_id", "mse_loss", "mae_loss"]],
            stdev=True,
        )
    return return_dict


def main():
    """
    if param.gpu_number = -1 it uses the cpu
    Returns:
    """
    # MODEL INIT
    ml_flow_run_id = None
    param = params_parser()
    console.log(f'Using the following params:{param}')
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if param.scale_level not in available_scale_levels.keys():
        raise ValueError(
            f"Oops!  That was no valid value for flag --scale_level. It has to be one of {available_scale_levels}")

    if param.dataset not in dataset_names.keys():
        raise ValueError(f"Oops!  That was no valid value for flag --dataset. It has to be be one of {dataset_names}")

    if param.use_ml_flow:
        mlflow.set_tracking_uri("http://localhost:9999")
        mlflow.set_experiment("ICRA 2022 experiments")
        mlflow.start_run()
        artifact_uri = mlflow.get_artifact_uri()
        console.log(f"artifact uri {artifact_uri}")

    patch_shape = (64, 64)
    image_shape = scaled_image_shapes[param.scale_level]

    model_save_folder_prefix, qualitative_paths, test_path, test_labels_csv, train_path, val_path, noise_path = uniformed_model_paths_init(
        param)
    if param.gpu_number == "-1":
        cuda_gpu = f"cpu"
    else:
        cuda_gpu = f"cuda:{param.gpu_number}"
    device = torch.device(cuda_gpu if torch.cuda.is_available() else "cpu")

    batch_size, bottleneck, epochs, input_channel, layer_1_ft, lr, max_patience, ml_flow_run_id, widths, workers = set_model_and_train_parameters_patches(
        device, ml_flow_run_id, param, patch_shape)

    train_patch_num = 1

    model = AE(widths, image_shape=patch_shape, bottleneck_size=bottleneck)
    model.to(device)
    if param.use_ml_flow:
        # noinspection PyUnboundLocalVariable
        model_save_folder = artifact_uri
    else:
        model_save_folder = model_save_folder_prefix + f'/saves/dataset_{dataset_names[param.dataset]}_B_{bottleneck}_F_{layer_1_ft}_S_{param.scale_level}_{time_string}'

    # AUG INIT

    composed_transform = A.Compose(
        [
            A.transforms.HorizontalFlip(p=0.5),
            A.transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomSizedCrop(min_max_height=[50, image_shape[0]], height=image_shape[0],
                              width=image_shape[1], p=0.5),
            A.Rotate(limit=10, p=0.5),
        ]
    )

    # DATA INIT
    train_set = UniformedPatchesDataset(train_path,
                                        patch_shape,
                                        max_patches=train_patch_num,
                                        aug_flag=True,
                                        transform=composed_transform,
                                        noise_path=noise_path,
                                        noise_flag=True,
                                        )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    with torch.no_grad():
        train_sample_batch = next(iter(train_loader))
        train_sample_batch = train_sample_batch.cpu()
        train_sample_batch_shape = train_sample_batch.shape
        train_sample_batch = train_sample_batch.view(train_sample_batch_shape[0] * train_sample_batch_shape[1],
                                                     train_sample_batch_shape[2],
                                                     train_sample_batch_shape[3],
                                                     train_sample_batch_shape[4])

    val_set = UniformedPatchesDataset(
        val_path,
        patch_shape,
        max_patches=train_patch_num,
        aug_flag=False,
        transform=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=max_patience, verbose=True)

    # TRAIN
    metrics = epoch_loop_patches(device, epochs, model, optimizer, param, scheduler, train_loader, val_loader)

    # TEST
    console.log("Training Completed, Testing Started")

    test_set = UniformedPatchesTestset(
        test_path,
        patch_shape=patch_shape,
        label_csv=test_labels_csv,
        max_patches=param.test_patches_number,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    with torch.no_grad():
        test_sample_batch = next(iter(test_loader))["patches"]
        test_sample_batch = test_sample_batch.cpu()
        test_sample_batch_shape = test_sample_batch.shape
        test_sample_batch = test_sample_batch.view(test_sample_batch_shape[0] * test_sample_batch_shape[1],
                                                   test_sample_batch_shape[2],
                                                   test_sample_batch_shape[3],
                                                   test_sample_batch_shape[4])
    df_dict = test_loop_df_rows(device, model, test_loader)
    test_set_df = pd.DataFrame.from_dict(df_dict)

    # Compute AUC fro ml flow logging

    uniformed_auc_computation_and_logging(test_set_df, param)

    # csv row building
    metrics_dict = {}
    for k, v in datasets_labels_names[param.dataset].items():
        auc_dict = per_label_metrics(test_set_df, k)
        if k == 0:
            v = "all"
        metrics_dict[f"{v}_mean_roc_auc_mse"] = auc_dict["mean_roc_auc"]["mse"]
        metrics_dict[f"{v}_mean_roc_auc_mae"] = auc_dict["mean_roc_auc"]["mae"]
        metrics_dict[f"{v}_mean_pr_auc_mse"] = auc_dict["mean_pr_auc"]["mse"]
        metrics_dict[f"{v}_mean_pr_auc_mae"] = auc_dict["mean_pr_auc"]["mae"]
        metrics_dict[f"{v}_q99_roc_auc_mse"] = auc_dict["q99_roc_auc"]["mse"]
        metrics_dict[f"{v}_q99_roc_auc_mae"] = auc_dict["q99_roc_auc"]["mae"]
        metrics_dict[f"{v}_q99_pr_auc_mse"] = auc_dict["q99_pr_auc"]["mse"]
        metrics_dict[f"{v}_q99_pr_auc_mae"] = auc_dict["q99_pr_auc"]["mae"]
        metrics_dict[f"{v}_std_roc_auc_mse"] = auc_dict["std_roc_auc"]["mse"]
        metrics_dict[f"{v}_std_roc_auc_mae"] = auc_dict["std_roc_auc"]["mae"]
        metrics_dict[f"{v}_std_pr_auc_mse"] = auc_dict["std_pr_auc"]["mse"]
        metrics_dict[f"{v}_std_pr_auc_mae"] = auc_dict["std_pr_auc"]["mae"]

    bfs_key = f"B{bottleneck}F{layer_1_ft}S{available_scale_levels[param.scale_level]}"
    csv_row = {
        **{"BFS": bfs_key,
           "dataset": param.dataset,
           "bottleneck": bottleneck,
           "first layer size": layer_1_ft,
           "scale level": available_scale_levels[param.scale_level],
           },
        **metrics_dict,
    }
    # SAVE STUFF
    console.log("Testing Completed")
    check_create_folder(model_save_folder)
    console.log("Creating and saving Artifacts")
    artifacts_path = model_save_folder + "/artifacts/"
    uniformed_model_artifact_saver(
        batch_size,
        bottleneck,
        epochs,
        layer_1_ft,
        lr,
        metrics,
        test_set_df,
        losses_list,
        model,
        input_channel,
        patch_shape,
        artifacts_path,
        ml_flow_run_id,
        param.id_optimized_loss,
        param=param,
        csv_row=csv_row,
        csv_key=f"d_{param.dataset}_{bfs_key}",
        train_sample_batch=train_sample_batch,
        test_sample_batch=test_sample_batch,
    )
    console.log(f"Script completed, artifacts located at {model_save_folder}."
                f" Videos and plots are skipped because still in todo ")


def uniformed_auc_computation_and_logging(test_df, param):
    label_unique_values = [0 if el[0] == 0 else 1 for el in
                           test_df[["frame_id", "frame_label"]].groupby("frame_id")["frame_label"].unique().values]
    metrics_dict = compute_uniformed_model_metrics(
        label_unique_values,
        losses_list, test_df[["frame_id", "mse_loss", "mae_loss"]])
    if param.use_ml_flow:
        mlflow.log_metric(f'test_set_q99_roc_auc_{losses_list[0]}', metrics_dict["q99_roc_auc"][losses_list[0]])
        mlflow.log_metric(f'test_set_q99_roc_auc_{losses_list[1]}', metrics_dict["q99_roc_auc"][losses_list[1]])
        mlflow.log_metric(f'test_set_mean_roc_auc_{losses_list[0]}', metrics_dict["mean_roc_auc"][losses_list[0]])
        mlflow.log_metric(f'test_set_mean_roc_auc_{losses_list[1]}', metrics_dict["mean_roc_auc"][losses_list[1]])
        mlflow.log_metric(f'test_set_q99_pr_auc_{losses_list[0]}', metrics_dict["q99_pr_auc"][losses_list[0]])
        mlflow.log_metric(f'test_set_q99_pr_auc_{losses_list[1]}', metrics_dict["q99_pr_auc"][losses_list[1]])
        mlflow.log_metric(f'test_set_mean_pr_auc_{losses_list[0]}', metrics_dict["mean_pr_auc"][losses_list[0]])
        mlflow.log_metric(f'test_set_mean_pr_auc_{losses_list[1]}', metrics_dict["mean_pr_auc"][losses_list[1]])
    else:
        print(f'test_set_q99_roc_auc_{losses_list[0]} = {metrics_dict["q99_roc_auc"][losses_list[0]]}\n'
              f'test_set_q99_roc_auc_{losses_list[1]} = {metrics_dict["q99_roc_auc"][losses_list[1]]}\n'
              f'test_set_mean_roc_auc_{losses_list[0]} = {metrics_dict["mean_roc_auc"][losses_list[0]]}\n'
              f'test_set_mean_roc_auc_{losses_list[1]} = {metrics_dict["mean_roc_auc"][losses_list[1]]}\n'
              f'test_set_q99_pr_auc_{losses_list[0]} = {metrics_dict["q99_pr_auc"][losses_list[0]]}\n'
              f'test_set_q99_pr_auc_{losses_list[1]} = {metrics_dict["q99_pr_auc"][losses_list[1]]}\n'
              f'test_set_mean_pr_auc_{losses_list[0]} = {metrics_dict["mean_pr_auc"][losses_list[0]]}\n'
              f'test_set_mean_pr_auc_{losses_list[1]} = {metrics_dict["mean_pr_auc"][losses_list[1]]}\n'
              )


if __name__ == "__main__":
    # with torch.autograd.detect_anomaly():
    #     main()
    main()
