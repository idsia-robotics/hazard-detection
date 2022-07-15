from utils.metrics_util import compute_oe_rnvp_model_metrics


# USED
def avg_l2_norms(test_df):
    ok_l2 = test_df.loc[test_df.label == 0].l2_norm_of_z.values.mean()
    an_l2 = test_df.loc[test_df.label != 0].l2_norm_of_z.values.mean()
    return an_l2, ok_l2


# USED
def rnvp_auc_computation_and_logging(test_df):
    labels = [0 if el == 0 else 1 for el in test_df["label"].values]
    metrics_dict = compute_oe_rnvp_model_metrics(
        labels,
        test_df
    )
    avg_an_l2_norm, avg_ok_l2_norm = avg_l2_norms(test_df)
    print(f'test_set_ok_mean_loss = {metrics_dict["ok_mean_loss"]}\n'
          f'test_set_an_mean_loss = {metrics_dict["an_mean_loss"]}\n'
          f'test_set_ok_mean_log_prob = {metrics_dict["ok_mean_log_prob"]}\n'
          f'test_set_an_mean_log_prob = {metrics_dict["an_mean_log_prob"]}\n'
          f'test_set_ok_mean_log_det_J = {metrics_dict["ok_mean_log_det_J"]}\n'
          f'test_set_an_mean_log_det_J = {metrics_dict["an_mean_log_det_J"]}\n'
          f'test_set_roc_auc = {metrics_dict["roc_auc"]}\n'
          f'test_set_pr_auc = {metrics_dict["pr_auc"]}\n'
          f'test_avg_an_l2_norm = {avg_an_l2_norm}\n'
          f'test_avg_ok_l2_norm = {avg_ok_l2_norm}\n'
          )


# USED
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
