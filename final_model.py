"""Final model: LDAGM branch (masked VGAE mu -> LDAGM) blended with a gradient-boosting stack, 5-fold CV x seeds.

Per fold (the training graph hides the test and validation positives; all choices use the validation pairs):
  LDAGM       ldagm_branch.ldagm_logits                                       (MIX-trained, early stopping)
  stack GBM   HistGradientBoosting on the 13 cross-fitted pair features of stack.py
  final       w z(LDAGM) + (1 - w) z(stack GBM), z = standardisation on the validation scores, w on a 0.1 grid
  degree rule log1p deg(l) + log1p deg(d) in the training graph (no learning; reference)
Tests: U (uniform negatives), D (degree-matched), H (heuristic) and MIX (1/3 each), all 1:1 with the test positives.
Output: AUC mean +- sd over seeds of the 5-fold mean, AUPR; differences vs final with a fold-stratified paired
bootstrap per seed; CSV in logs/final_model/summary_<split>.csv.

Usage (repo root):  python final_model.py <split | all> [n_seeds]       e.g.  python final_model.py dataset1_v2026_leaf
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import obonet
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

import config
import data
import stack
from build_eval_splits import DOID_OBO, ancestor_matrix
from ldagm_branch import CACHE, ldagm_logits, logit

torch.set_num_threads(1)
MODELS = ["degree rule", "LDAGM", "stack GBM", "final"]
TESTS = ("U", "D", "H", "MIX")


def zfit(s_val):
    m, sd = s_val.mean(), s_val.std() + 1e-12
    return lambda s: (s - m) / sd


def aupr(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return auc(r, p)


def run_fold(dataset, fold, s, ctx, cache=CACHE):
    """{"y": test labels, "s": {model: {test: scores}}, "w": blend weight, "gbm_iter": boosting iterations}."""
    path = f"{cache}/final_{dataset}_s{s}_f{fold + 1}.npz"
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)["rec"].item()
    sets = data.fold_sets(dataset, fold, ctx["negD"])
    tests = [t for t in TESTS if t in sets]
    x = data.fold_inputs(dataset, fold)
    n_lnc = x["lnc_di"].shape[0]
    pairs = {k: np.vstack([p, n]) for k, (p, n) in sets.items()}
    labels = {k: np.r_[np.ones(len(p)), np.zeros(len(n))] for k, (p, n) in sets.items()}
    yv = labels["val"]
    L = ldagm_logits(dataset, fold, s, sets, x, ["val"] + tests, cache)

    Y = (x["lnc_di"] > 0).astype(float)
    SEM, LM, MD = x["di_semantic"].astype(float), x["lnc_mi"].astype(float), x["mi_di"].astype(float)
    LJ = data.mirna_jaccard(LM)
    get = lambda Fm, pr: Fm[pr[:, 0], pr[:, 1] - n_lnc]
    sig = stack.choose_sigmas(Y, SEM, pairs["val"], yv)
    full = stack.feature_maps(Y, SEM, LM, MD, LJ, sig)
    X_fit, used = stack.cross_fitted_features(dataset, Y, SEM, LM, MD, LJ, sig, pairs["fit"], labels["fit"], ctx["cols"],
                                              ctx["G"], ctx["A"], np.random.default_rng(config.SEED + 7 * fold + 1000 * s))
    g = HistGradientBoostingClassifier(**config.GBM, random_state=config.SEED + fold + 1000 * s)
    g.fit(X_fit[used], labels["fit"][used])
    S = {k: logit(g.predict_proba(get(full, pairs[k]))[:, 1]) for k in ["val"] + tests}

    za, zb = zfit(L["val"]), zfit(S["val"])
    grid = np.round(np.linspace(0, 1, config.BLEND_GRID), 2)
    w = max(grid, key=lambda w: (roc_auc_score(yv, w * za(L["val"]) + (1 - w) * zb(S["val"])), w))
    dl, dd = np.log1p(Y.sum(1)), np.log1p(Y.sum(0))
    rec = {"y": {t: labels[t] for t in tests},
           "s": {"degree rule": {t: dl[pairs[t][:, 0]] + dd[pairs[t][:, 1] - n_lnc] for t in tests},
                 "LDAGM": {t: L[t] for t in tests},
                 "stack GBM": {t: S[t] for t in tests},
                 "final": {t: w * za(L[t]) + (1 - w) * zb(S[t]) for t in tests}},
           "w": float(w), "gbm_iter": int(g.n_iter_)}
    os.makedirs(cache, exist_ok=True)
    np.savez(path, rec=np.array(rec, dtype=object))
    return rec


def context(dataset):
    """Per-split inputs shared by the folds: D negatives, disease columns, Disease Ontology, ancestor matrix (_leaf)."""
    cols = list(pd.read_csv(f"./our_dataset/{dataset}/interaction/lnc_di.csv", index_col="0").columns)
    G = obonet.read_obo(DOID_OBO) if dataset.endswith(("_fam", "_leaf")) else None
    return {"negD": data.degree_matched_negatives(dataset), "cols": cols, "G": G,
            "A": ancestor_matrix(cols, G) if dataset.endswith("_leaf") else None}


def boot(folds, a, b, t, B=300, seed=0):
    """Mean over folds of AUC(a) - AUC(b) on test t, with a 95% paired bootstrap interval (pairs resampled per fold)."""
    rng = np.random.default_rng(seed)
    obs = np.mean([roc_auc_score(f["y"][t], f["s"][a][t]) - roc_auc_score(f["y"][t], f["s"][b][t]) for f in folds])
    ds = []
    for _ in range(B):
        v = []
        for f in folds:
            y = f["y"][t]
            i = rng.integers(0, len(y), len(y))
            v.append(roc_auc_score(y[i], f["s"][a][t][i]) - roc_auc_score(y[i], f["s"][b][t][i]))
        ds.append(np.mean(v))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return obs, lo, hi


def summarise(dataset, runs):
    tests = list(runs[0][0]["y"])
    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    rows = []
    print(f"\n=== {dataset} ({config.SPLITS.get(dataset, '')}): {len(runs)} seed(s) | AUC mean +- sd over seeds of the "
          f"5-fold mean (AUPR mean) ===")
    for m in MODELS:
        cells = []
        for t in tests:
            a = [np.mean([roc_auc_score(f["y"][t], f["s"][m][t]) for f in folds]) for folds in runs]
            p = [np.mean([aupr(f["y"][t], f["s"][m][t]) for f in folds]) for folds in runs]
            rows.append({"split": dataset, "model": m, "test": t, "auc_mean": np.mean(a), "auc_sd": sd(a),
                         "aupr_mean": np.mean(p)})
            cells.append(f"{t} {np.mean(a):.4f}+-{sd(a):.4f} ({np.mean(p):.4f})")
        print(f"{m:11s} | " + " | ".join(cells))
    print("--- final - X in AUC points: mean +- sd over seeds (seeds whose 95% CI excludes 0: positive / negative) ---")
    for b_ in MODELS[:-1]:
        cells = []
        for t in tests:
            bs = [boot(folds, "final", b_, t, seed=k) for k, folds in enumerate(runs)]
            d = [100 * b[0] for b in bs]
            cells.append(f"{t} {np.mean(d):+.2f}+-{sd(d):.2f} ({sum(b[1] > 0 for b in bs)}+/{sum(b[2] < 0 for b in bs)}-)")
        print(f"final - {b_:11s}: " + " | ".join(cells))
    fl = [f for folds in runs for f in folds]
    print(f"mean blend weight w on LDAGM {np.mean([f['w'] for f in fl]):.2f} | GBM iterations "
          f"{np.mean([f['gbm_iter'] for f in fl]):.0f} (cap {config.GBM['max_iter']})", flush=True)
    os.makedirs("logs/final_model", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"logs/final_model/summary_{dataset}.csv", index=False)


def run(dataset, n_seeds=config.N_SEEDS):
    t0 = time.time()
    ctx = context(dataset)
    runs = []
    for s in range(n_seeds):
        runs.append([run_fold(dataset, f, s, ctx) for f in range(config.TOTAL_FOLDS)])
        print(f"  [{dataset}] seed {s} done | w per fold {[f['w'] for f in runs[-1]]} | {time.time() - t0:.0f}s", flush=True)
    summarise(dataset, runs)
    print(f"(done in {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    splits = list(config.SPLITS) if sys.argv[1] == "all" else [sys.argv[1]]
    for ds in splits:
        run(ds, int(sys.argv[2]) if len(sys.argv) > 2 else config.N_SEEDS)
