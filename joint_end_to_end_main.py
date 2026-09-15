import time
import warnings

import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)

import config
from experiment_logging import log_hyperparameters, log_to_csv
from fold_data_loader import load_fold_data
from joint_train_eval import joint_train, joint_test

warnings.filterwarnings("ignore")

METRIC_NAMES = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']


def select_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.DEVICE in ["cuda", "mps"]:
        print(f"Warning: {config.DEVICE} not available, falling back to CPU")
    return torch.device("cpu")


def evaluate_predictions(test_labels, test_predictions):
    """Compute AUC/AUPR/MCC/ACC/Precision/Recall/F1 from raw prediction scores."""
    AUC = roc_auc_score(test_labels, test_predictions)
    precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
    AUPR = auc(recall, precision)

    binary_preds = (test_predictions > 0.5).astype(int)
    MCC = matthews_corrcoef(test_labels, binary_preds)
    ACC = accuracy_score(test_labels, binary_preds)
    P = precision_score(test_labels, binary_preds)
    R = recall_score(test_labels, binary_preds)
    F1 = f1_score(test_labels, binary_preds)

    return {"AUC": AUC, "AUPR": AUPR, "MCC": MCC, "ACC": ACC, "Precision": P, "Recall": R, "F1-Score": F1}


def run_fold(fold, dataset, device):
    """Train and evaluate the joint model for a single cross-validation fold."""
    print(f"\n{'='*50}")
    print(f"Starting Fold {fold + 1}/5")
    print(f"{'='*50}")

    fold_data = load_fold_data(dataset, fold, device)

    print(f"Starting joint end-to-end training for fold {fold + 1}...")
    trained_model, _ = joint_train(
        num_lnc=fold_data.num_lnc,
        num_diseases=fold_data.num_diseases,
        num_mi=fold_data.num_mi,
        train_dataset=fold_data.train_dataset,
        multi_view_data=fold_data.multi_view_data,
        lnc_di_interaction=fold_data.lnc_di_tensor,
        lnc_mi_interaction=fold_data.lnc_mi_tensor,
        mi_di_interaction=fold_data.mi_di_tensor,
        fold=fold,
        device=device
    )

    print(f"\nTesting joint model for fold {fold + 1}...")
    test_labels, test_predictions = joint_test(
        model=trained_model,
        test_dataset=fold_data.test_dataset,
        multi_view_data=fold_data.multi_view_data,
        lnc_di_interaction=fold_data.lnc_di_tensor,
        lnc_mi_interaction=fold_data.lnc_mi_tensor,
        mi_di_interaction=fold_data.mi_di_tensor,
        fold=fold,
        batch_size=config.BATCH_SIZE,
        device=device
    )

    metrics = evaluate_predictions(test_labels, test_predictions)

    print(f"\n=== Fold {fold + 1} Results ===")
    for name in METRIC_NAMES:
        print(f"{name}: {metrics[name]:.4f}")

    return {"fold": fold + 1, **metrics}


def format_runtime(total_seconds):
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s


def summarize_results(all_fold_results, start_time):
    """Print, log, and CSV-log the mean/std of metrics across all folds."""
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)

    results_df = pd.DataFrame(all_fold_results)

    print("\nDetailed Results by Fold:")
    print(results_df.to_string(index=False, float_format='%.4f'))

    print("\nStatistical Summary:")
    print("-" * 50)
    for metric in METRIC_NAMES:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric:12s}: {mean_val:.4f} ± {std_val:.4f}")

    h, m, s = format_runtime(time.time() - start_time)
    print(f"\nTotal runtime: {h} hours, {m} minutes, and {s:.2f} seconds")

    with open(config.LOG_FILE, 'a') as f:
        f.write("\n=== 5-Fold Cross-Validation Results ===\n")
        for metric in METRIC_NAMES:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write(f"Total runtime: {h} hours, {m} minutes, and {s:.2f} seconds\n")
        f.write("================================================\n")

    avg_metrics = {metric: results_df[metric].mean() for metric in METRIC_NAMES}
    avg_metrics["Time"] = f"{h} hours, {m} minutes, and {s:.2f} seconds"
    log_to_csv(config, avg_metrics)


def main():
    log_hyperparameters(config)
    start_time = time.time()

    dataset = config.DATASET
    device = select_device()
    print(f"Using device: {device}")

    print(f"Starting 5-fold cross-validation for {dataset}...")

    all_fold_results = [
        run_fold(fold, dataset, device) for fold in range(config.TOTAL_FOLDS)
    ]

    summarize_results(all_fold_results, start_time)

    print("\n5-fold cross-validation completed successfully!")


if __name__ == '__main__':
    main()
