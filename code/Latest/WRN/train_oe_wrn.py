import argparse
from collections import defaultdict
from datetime import datetime
from typing import List

from models.wrn_oe.wrn import WideResNet
from utils.variables_util import combined_labels_to_names
from utils.metrics_util import compute_pr_aucs_single_loss, compute_roc_aucs_single_loss
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
from rich.console import Console
from tqdm import tqdm

from models.wrn_oe.wrn_model_functions import set_wrn_class_parameters, \
    cosine_annealing, train_wrn, val_wrn, best_epoch_saver, test_wrn, oe_wrn_artifact_saver
from oe_wrn_dataset import OEWRNImagesDataset, OEWRNImagesTestset, \
    OEWRNImagesOutlierset
from utils.check_create_folder import check_create_folder

console = Console()


def main():
    parser = argparse.ArgumentParser(description='Trains a WRT with OE',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, default='wrn',
                        choices=['allconv', 'wrn'], help='Choose architecture.')
    # parser.add_argument('--calibration', '-c', action='store_true',
    #                     help='Train a model to be used for calibration. This holds out some data for validation.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    # WRN Architecture
    parser.add_argument('--layers', default=16, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=4, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    # parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--workers', type=int, default=4, help='Pre-fetching threads.')
    # IDSIA params
    parser.add_argument('--gpu_number',
                        '-g', type=int, default=0)
    parser.add_argument('--root_path',
                        '-r', type=str, default=".")
    param = parser.parse_args()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(64, padding=8),
                                   trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.Normalize(mean, std)])

    num_classes = 3  # the classes are the envs: -1, 1long, 1short

    console.log(f'Using the following params:{param}')
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    if param.gpu_number == -1:
        cuda_gpu = f"cpu"
    else:
        cuda_gpu = f"cuda:{param.gpu_number}"
    # (
    #     folder_prefix,
    #     train_set_path,
    #     outliers_set_path,
    #     validation_set_path,
    #     test_set_path,
    #     test_set_labels_csv,
    #     outliers_set_labels_csv,
    #     train_set_labels_csv,
    #     validation_set_labels_csv
    # ) = oe_wrn_model_paths_init(param)
    root_path = param.root_path

    train_set_path = f"{root_path}/data/train_set"
    outliers_set_path = f"{root_path}/data/outlier_set"
    validation_set_path = f"{root_path}/data/validation_set"
    test_set_path = f"{root_path}/data/test_set"
    test_set_labels_csv = f"{root_path}/data/metadata/frames_labels.csv"
    outliers_set_labels_csv = f"{root_path}/data/metadata/outliers_frames_labels.csv"
    train_set_labels_csv = f"{root_path}/data/metadata/train_frames_envs.csv"
    validation_set_labels_csv = f"{root_path}/data/metadata/validation_frames_envs.csv"

    model_key = f"WRN_{time_string}"
    save_folder = f"{root_path}/data/wrn/saves/{model_key}"
    check_create_folder(save_folder)

    device = torch.device(cuda_gpu if torch.cuda.is_available() else "cpu")
    print(device)
    (
        batch_size,
        epochs,
        lr,
        ml_flow_run_id,
        workers,
        momentum,
        decay,
        layers,
        widen_factor,
        droprate,
        outlier_batch_size,
        analysis_number,
        split_date,
        subset_size
    ) = set_wrn_class_parameters(
        param
    )

    model = WideResNet(layers, num_classes, widen_factor, dropRate=droprate)
    model.to(device)
    cudnn.benchmark = True  # fire on all cylinders

    # DATA INIT
    train_set = OEWRNImagesDataset(train_set_path, train_transform, train_set_labels_csv)
    train_loader_in = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    outliers_set_required_size = outlier_batch_size * len(train_loader_in)
    outliers_set = OEWRNImagesOutlierset(
        files_path=outliers_set_path,
        label_csv_path=outliers_set_labels_csv,
        transform=test_transform,
        required_dataset_size=outliers_set_required_size,
    )
    train_loader_out = torch.utils.data.DataLoader(
        outliers_set,
        batch_size=outlier_batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    val_set = OEWRNImagesDataset(validation_set_path, test_transform, validation_set_labels_csv)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    test_set = OEWRNImagesTestset(test_set_path, label_csv_path=test_set_labels_csv, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True)
    print('Beginning Training\n')
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / lr))
    # Main loop
    etqdm = tqdm(range(epochs), total=epochs, postfix="Training")
    metrics = defaultdict(list)
    best_loss = None
    best_epoch = 0
    best_model_path = None
    for epoch in etqdm:
        model, train_loss = train_wrn(model, train_loader_out, train_loader_in, scheduler, optimizer, device=device)
        metrics[f'train_loss'].append(train_loss)
        val_loss = val_wrn(model, val_loader, device)
        metrics[f'val_loss'].append(val_loss)
        if best_loss is None:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model, param,
                                                                      save_folder, val_loss, model_key)
        if val_loss < best_loss:
            best_epoch, best_loss, best_model_path = best_epoch_saver(epoch, model, param,
                                                                      save_folder, val_loss, model_key)

        etqdm.set_description(
            f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | best model @ epoch {best_epoch}")

    console.log("Training Completed, Testing Started")

    df_dict = test_wrn(best_model_path, model, test_loader, device, )
    test_set_df = pd.DataFrame.from_dict(df_dict)
    test_set_df["label"] = pd.to_numeric(test_set_df["label"])

    auc_computation_and_logging(test_set_df)
    list_of_labels_in_test_set = list(set(test_set.labels))

    # csv row building
    metrics_dict = {}
    for k in list_of_labels_in_test_set:
        v = combined_labels_to_names[k]
        class_metrics_dict = per_label_metrics(test_set_df, k)
        if k == 0:
            v = "all"
            metrics_dict[f"{v}_ok_mean_loss"] = class_metrics_dict["ok_mean_loss"]
        metrics_dict[f"{v}_an_mean_loss"] = class_metrics_dict["an_mean_loss"]
        metrics_dict[f"{v}_roc_auc"] = class_metrics_dict["roc_auc"]
        metrics_dict[f"{v}_pr_auc"] = class_metrics_dict["pr_auc"]

    csv_row = {
        **{"model_key": model_key,
           },
        **metrics_dict,
    }

    # SAVE STUFF
    console.log("Testing Completed")
    console.log("Creating and saving Artifacts")
    artifacts_path = save_folder + "/artifacts/"
    oe_wrn_artifact_saver(
        batch_size,
        epochs,
        lr,
        momentum,
        decay,
        layers,
        widen_factor,
        droprate,
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


def auc_computation_and_logging(test_df):
    labels = [0 if el == 0 else 1 for el in test_df["label"].values]
    metrics_dict = compute_model_metrics(
        labels,
        test_df
    )
    print(f'test_set_ok_mean_loss = {metrics_dict["ok_mean_loss"]}\n'
          f'test_set_an_mean_loss = {metrics_dict["an_mean_loss"]}\n'
          f'test_set_roc_auc = {metrics_dict["roc_auc"]}\n'
          f'test_set_pr_auc = {metrics_dict["pr_auc"]}\n'
          )


def per_label_metrics(df, label_key):
    if label_key == 0:
        label_unique_values = [0 if el == 0 else 1 for el in df["label"].values]
        return_dict = compute_model_metrics(
            label_unique_values,
            df,
        )
    else:
        df_anomaly = df[df.label.isin([0, label_key])]
        label_unique_values = [0 if el == 0 else 1 for el in df_anomaly["label"].values]
        return_dict = compute_model_metrics(
            label_unique_values,
            df_anomaly,
        )
    return return_dict


def compute_model_metrics(
        labels: List[int],
        df_losses: pd.DataFrame,
):
    y_true = labels
    losses = df_losses["loss"].values
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


if __name__ == '__main__':
    main()
