"""Two-stage pipeline, stage 2: LDAGM on the fixed node embeddings, with the same folds, validation split, early
stopping and metrics as joint_end_to_end_main.py.

Node features (concat8): x_i = [mu^(1)_i || ... || mu^(4)_i || R_e^(1)_i || ... || R_e^(4)_i]  (8 x 128 = 1024),
from build_offline_features.py, every embedding matrix z-scored per column over all nodes (no labels involved).
Pair features: [x_lncRNA || x_disease] -> LDAGM -> association score.

Usage: python build_offline_features.py --dataset dataset2      (once per dataset)
       python fixed_feature_main.py --dataset dataset2
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config
import fold_data_loader as fdl
from build_offline_features import feature_dir
from joint_dataset import JointDataset
from joint_end_to_end_main import METRIC_NAMES, evaluate_predictions
from LDAGM import LDAGM

RESULTS_CSV = "logs/two_stage_results.csv"


def _zscore(X):
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - X.mean(axis=0)) / sd


def _load_stack(folder, fold, num_stages=config.NETWORK_NUM):
    return [np.load(os.path.join(folder, f"A_{fold + 1}_{k}.npy")) for k in range(1, num_stages + 1)]


def node_features(dataset, fold):
    """(N, 8 * OFFLINE_EMBED_DIM) float32 concat8 node features of one fold: mu^(1..4) then R_e^(1..4)."""
    folder = feature_dir(dataset)
    mus = [_zscore(m) for m in _load_stack(os.path.join(folder, "VGAE_encoder"), fold)]
    res = [_zscore(r) for r in _load_stack(os.path.join(folder, "A_encoder"), fold)]
    return np.concatenate(mus + res, axis=1).astype(np.float32)


def fold_datasets(dataset, fold):
    """Train / validation / test pairs, identical to fold_data_loader.load_fold_data."""
    (_, _, train_pos, train_neg, test_pos, test_neg) = fdl._load_fold_split_indices(dataset, fold)
    train_pos, train_neg, val_pos, val_neg = fdl._split_validation(
        train_pos, train_neg, config.VAL_RATIO, config.SEED + fold)
    return (JointDataset(train_pos, train_neg, "train", dataset),
            JointDataset(val_pos, val_neg, "val", dataset),
            JointDataset(test_pos, test_neg, "test", dataset))


def _pair_features(X, pairs):
    return torch.cat([X[pairs[:, 0]], X[pairs[:, 1]]], dim=1)


def _score(model, X, ds):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(_pair_features(X, ds.node_pairs))).numpy()
    return ds.labels.numpy(), p


def train_ldagm(X, train_ds, val_ds):
    """LDAGM on fixed features with early stopping on validation AUC; returns (model, best_epoch, best_val_auc)."""
    model = LDAGM(input_dimension=2 * X.shape[1], hidden_dimension=config.LDAGM_HIDDEN_DIM, feature_num=1,
                  hiddenLayer_num=config.LDAGM_LAYERS, drop_rate=config.DROP_RATE, use_aggregate=config.USE_AGGREGATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    best_auc, best_epoch, best_state, waited = -np.inf, 0, None, 0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for pairs, labels in loader:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(_pair_features(X, pairs)), labels)
            loss.backward()
            optimizer.step()
        val_auc = roc_auc_score(*_score(model, X, val_ds))
        if val_auc > best_auc:
            best_auc, best_epoch, waited = val_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config.EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_auc


def run_experiment(dataset, verbose=True):
    """5-fold CV of LDAGM on the concat8 features. Returns a summary dict (also appended to RESULTS_CSV);
    summary["_folds"] holds the per-fold rows (not written to the CSV)."""
    t0 = time.time()
    tag, _ = config.PPR_ALPHAS[dataset]
    rows = []
    for fold in range(config.TOTAL_FOLDS):
        seed = config.SEED + fold
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        X = torch.tensor(node_features(dataset, fold))
        train_ds, val_ds, test_ds = fold_datasets(dataset, fold)
        model, best_epoch, val_auc = train_ldagm(X, train_ds, val_ds)
        labels, preds = _score(model, X, test_ds)
        metrics = evaluate_predictions(labels, preds)
        rows.append({"fold": fold + 1, "val_auc": val_auc, "best_epoch": best_epoch, **metrics})
        if verbose:
            print(f"  [{dataset} {tag}] fold {fold + 1}: val AUC {val_auc:.4f} | test AUC {metrics['AUC']:.4f} "
                  f"AUPR {metrics['AUPR']:.4f} | best epoch {best_epoch}", flush=True)
    summary = {"dataset": dataset, "tag": tag, "node_dim": int(X.shape[1]),
               "val_auc": float(np.mean([r["val_auc"] for r in rows]))}
    for m in METRIC_NAMES:
        vals = np.array([r[m] for r in rows])
        summary[m], summary[m + "_std"] = float(vals.mean()), float(vals.std(ddof=1))
    summary["best_epochs"] = " ".join(str(r["best_epoch"]) for r in rows)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time"] + list(summary))
        if new:
            writer.writeheader()
        writer.writerow({"time": time.strftime("%Y-%m-%d %H:%M"), **summary})
    if verbose:
        print(f"  => {dataset} {tag}: val AUC {summary['val_auc']:.4f} | test AUC {summary['AUC']:.4f} ± "
              f"{summary['AUC_std']:.4f}, AUPR {summary['AUPR']:.4f} ± {summary['AUPR_std']:.4f} "
              f"({summary['minutes']} min)", flush=True)
    summary["_folds"] = rows
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=config.DATASET, choices=sorted(config.PPR_ALPHAS))
    args = parser.parse_args()
    run_experiment(args.dataset)
