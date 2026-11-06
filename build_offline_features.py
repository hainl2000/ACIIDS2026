"""Two-stage pipeline, stage 1: build the fixed node embeddings of one dataset.

For every fold f = 1..5 and PPR stage k = 1..4 (stage_graphs.build_stage_graphs with the alphas of
config.PPR_ALPHAS[dataset]), with the test AND validation positives hidden, it trains
    GCN-AE   (input and target A_k, MSE)                   -> R_e^(k)   offline/<tag>/A_encoder/A_<f>_<k>.npy
    VGAE-MLP (target: association blocks of A_k, BCE + KL)  -> mu^(k)    offline/<tag>/VGAE_encoder/A_<f>_<k>.npy
under <root>/<dataset>/, and writes offline/<tag>/meta.json (settings + per-model diagnostics) once all folds are
done. <tag> names the alpha set, e.g. a50-30-15-05.

Usage: python build_offline_features.py --dataset dataset2
"""
import argparse
import json
import os
import time

import numpy as np

import config
from offline_encoders import association_mask, train_gcn_ae, train_vgae_mlp
from stage_graphs import build_stage_graphs


def feature_dir(dataset, root="./our_dataset"):
    tag, _ = config.PPR_ALPHAS[dataset]
    return os.path.join(root, dataset, "offline", tag)


def build_features(dataset, folds=None, root="./our_dataset", overwrite=False):
    """Train all offline encoders of a dataset; returns the meta dict."""
    tag, alphas = config.PPR_ALPHAS[dataset]
    out = feature_dir(dataset, root)
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path) and not overwrite:
        print(f"[build] {meta_path} exists, skipping", flush=True)
        with open(meta_path) as f:
            return json.load(f)
    folds = list(range(config.TOTAL_FOLDS)) if folds is None else folds
    os.makedirs(os.path.join(out, "A_encoder"), exist_ok=True)
    os.makedirs(os.path.join(out, "VGAE_encoder"), exist_ok=True)
    meta = {
        "dataset": dataset, "tag": tag, "alphas": list(alphas),
        "hidden_positives": "test + validation", "folds": [f + 1 for f in folds],
        "embed_dim": config.OFFLINE_EMBED_DIM,
        "gcn_ae": {"lr": config.OFFLINE_AE_LR, "weight_decay": config.OFFLINE_AE_WEIGHT_DECAY, "epochs": config.OFFLINE_AE_EPOCHS},
        "vgae_mlp": {"hidden": config.OFFLINE_VGAE_HIDDEN_DIM, "lr": config.OFFLINE_VGAE_LR,
                     "epochs": config.OFFLINE_VGAE_EPOCHS, "kl_weight": config.OFFLINE_VGAE_KL_WEIGHT,
                     "target": "association blocks of A_k"},
        "models": [],
    }
    for fold in folds:
        t0 = time.time()
        graphs, (n_lnc, n_dis, n_mi) = build_stage_graphs(dataset, fold, alphas)
        mask = association_mask(n_lnc, n_dis, n_mi)
        for k, A in enumerate(graphs, start=1):
            seed = config.SEED + 100 * fold + k
            re, ae_mse = train_gcn_ae(A, config.OFFLINE_EMBED_DIM, config.OFFLINE_AE_LR,
                                      config.OFFLINE_AE_WEIGHT_DECAY, config.OFFLINE_AE_EPOCHS, seed)
            np.save(os.path.join(out, "A_encoder", f"A_{fold + 1}_{k}.npy"), re.astype(np.float32))
            mu, info = train_vgae_mlp(A, mask, config.OFFLINE_VGAE_HIDDEN_DIM, config.OFFLINE_EMBED_DIM,
                                      config.OFFLINE_VGAE_LR, config.OFFLINE_VGAE_EPOCHS,
                                      config.OFFLINE_VGAE_KL_WEIGHT, seed)
            np.save(os.path.join(out, "VGAE_encoder", f"A_{fold + 1}_{k}.npy"), mu.astype(np.float32))
            entry = {"fold": fold + 1, "stage": k, "alpha": alphas[k - 1], "ae_mse": ae_mse,
                     "ae_live_dims": int((~(re == 0).all(0)).sum()), **{f"vgae_{a}": b for a, b in info.items()}}
            meta["models"].append(entry)
            print(f"[build] {dataset} {tag} fold {fold + 1} stage {k}: AE mse {ae_mse:.4f}, live dims "
                  f"{entry['ae_live_dims']} | VGAE recon {info['recon']:.4f} kl {info['kl']:.3f} "
                  f"train AUC {info['train_recon_auc']:.4f}", flush=True)
        print(f"[build] {dataset} {tag} fold {fold + 1} done in {time.time() - t0:.0f}s", flush=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=config.DATASET, choices=sorted(config.PPR_ALPHAS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_features(args.dataset, overwrite=args.overwrite)
