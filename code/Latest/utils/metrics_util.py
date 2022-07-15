from typing import List, Dict

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


# USED
def compute_oe_rnvp_model_metrics(
        labels: List[int],
        df_losses: pd.DataFrame,
):
    y_true = labels
    losses = df_losses["loss"].values
    pr_auc = compute_pr_aucs_single_loss(y_true, losses)
    roc_auc = compute_roc_aucs_single_loss(y_true, losses)

    an_mean_loss = df_losses[df_losses["label"] != 0]["loss"].values.mean()
    an_mean_prob = df_losses[df_losses["label"] != 0]["log_prob"].values.mean()
    an_mean_detj = df_losses[df_losses["label"] != 0]["log_det_J"].values.mean()
    an_mean_l2_norm_of_z = df_losses[df_losses["label"] != 0]["l2_norm_of_z"].values.mean()

    ok_mean_loss = df_losses[df_losses["label"] == 0]["loss"].values.mean()
    ok_mean_prob = df_losses[df_losses["label"] == 0]["log_prob"].values.mean()
    ok_mean_detj = df_losses[df_losses["label"] == 0]["log_det_J"].values.mean()
    ok_mean_l2_norm_of_z = df_losses[df_losses["label"] == 0]["l2_norm_of_z"].values.mean()
    # composing return dict
    return_dict = {
        "an_mean_loss": an_mean_loss,
        "an_mean_log_prob": an_mean_prob,
        "an_mean_log_det_J": an_mean_detj,
        "an_mean_l2_norm_of_z": an_mean_l2_norm_of_z,
        "ok_mean_loss": ok_mean_loss,
        "ok_mean_log_prob": ok_mean_prob,
        "ok_mean_log_det_J": ok_mean_detj,
        "ok_mean_l2_norm_of_z": ok_mean_l2_norm_of_z,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    return return_dict


# USED
def compute_pr_aucs(y_true: List[int], losses_list: List[str], test_set_losses: Dict[str, float]):
    dict_aucs = {}
    for loss in losses_list:
        y_score = test_set_losses[loss]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        dict_aucs[loss] = pr_auc
    return dict_aucs


# USED
def compute_roc_aucs(y_true: List[int], losses_list: List[str], test_set_losses: Dict[str, float]):
    dict_aucs = {}
    for loss in losses_list:
        y_score = test_set_losses[loss]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        dict_aucs[loss] = roc_auc
    return dict_aucs


# USED
def compute_pr_aucs_single_loss(y_true: List[int], y_score: Dict[str, float]):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return pr_auc


# USED
def compute_roc_aucs_single_loss(y_true: List[int], y_score: List[float]):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc
