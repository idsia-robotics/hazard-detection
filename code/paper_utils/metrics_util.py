from typing import List, Dict

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def uniformed_model_group_losses_by_group_method_and_loss_name(df_losses, loss_name, grouping_method):
    if grouping_method == "q99":
        return df_losses[["frame_id", f"{loss_name}_loss"]].groupby("frame_id").quantile(q=0.99).values
    elif grouping_method == "mean":
        return df_losses[["frame_id", f"{loss_name}_loss"]].groupby("frame_id").mean().values
    elif grouping_method == "std":
        return df_losses[["frame_id", f"{loss_name}_loss"]].groupby("frame_id").std(ddof=0).values
    else:
        raise Exception("Grouping method not found")


def compute_uniformed_model_metrics(
        labels: List[int],
        list_losses: List[str],
        df_losses: pd.DataFrame,
        stdev=False,
):
    # this code produces the q99 and mean of N patches X ROC auc and PR AUC X mse, mae = 8 metrics

    y_test = labels

    # q99

    quant_test_set_losses = {
        list_losses[0]: uniformed_model_group_losses_by_group_method_and_loss_name(df_losses, list_losses[0], "q99"),
        list_losses[1]: uniformed_model_group_losses_by_group_method_and_loss_name(df_losses, list_losses[1], "q99"),
    }
    q99_pr_auc = compute_pr_aucs(y_test, list_losses, quant_test_set_losses)
    q99_roc_auc = compute_roc_aucs(y_test, list_losses, quant_test_set_losses)

    # mean
    mean_test_set_losses = {
        list_losses[0]: uniformed_model_group_losses_by_group_method_and_loss_name(df_losses, list_losses[0], "mean"),
        list_losses[1]: uniformed_model_group_losses_by_group_method_and_loss_name(df_losses, list_losses[1], "mean"),
    }
    mean_pr_auc = compute_pr_aucs(y_test, list_losses, mean_test_set_losses)
    mean_roc_auc = compute_roc_aucs(y_test, list_losses, mean_test_set_losses)

    if not stdev:
        # composing return dict
        return_dict = {
            "q99_roc_auc": q99_roc_auc,
            "q99_pr_auc": q99_pr_auc,
            "mean_roc_auc": mean_roc_auc,
            "mean_pr_auc": mean_pr_auc,
        }
    else:
        # std
        std_test_set_losses = {
            list_losses[0]: uniformed_model_group_losses_by_group_method_and_loss_name(df_losses, "mse", "std"),
            list_losses[1]: uniformed_model_group_losses_by_group_method_and_loss_name(df_losses, "mae", "std"),
        }
        std_pr_auc = compute_pr_aucs(y_test, list_losses, std_test_set_losses)
        std_roc_auc = compute_roc_aucs(y_test, list_losses, std_test_set_losses)

        # composing return dict
        return_dict = {
            "q99_roc_auc": q99_roc_auc,
            "q99_pr_auc": q99_pr_auc,
            "mean_roc_auc": mean_roc_auc,
            "mean_pr_auc": mean_pr_auc,
            "std_roc_auc": std_roc_auc,
            "std_pr_auc": std_pr_auc,
        }
    return return_dict


def compute_pr_aucs(y_test: List[int], list_losses: List[str], test_set_losses: Dict[str, float]):
    dict_aucs = {}
    for loss in list_losses:
        y_score = test_set_losses[loss]
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)
        dict_aucs[loss] = pr_auc
    return dict_aucs


def compute_roc_aucs(y_test: List[int], list_losses: List[str], test_set_losses: Dict[str, float]):
    dict_aucs = {}
    for loss in list_losses:
        y_score = test_set_losses[loss]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        dict_aucs[loss] = roc_auc
    return dict_aucs
