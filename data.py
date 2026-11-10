"""Data of the final model: interaction matrices, folds, validation split, test negatives and the training graph of a fold.

our_dataset/<split>/ (built by build_updated_datasets.py -> data_preprocessing.ipynb -> build_eval_splits.py):
  interaction/         lnc_di / lnc_mi / mi_di CSVs (index column '0')
  index/               positive_ij / negative_ij (disease column offset by n_lnc), positive5foldsidx / negative5foldsidx
                       (object arrays), optional val5foldsidx (group-aware validation), negD_ij (degree-matched
                       negatives, aligned with positive_ij), negH_fold<k> (heuristic test negatives)
  multi_similarities/  di_semantic, mi_func, mi_gip similarities (fold-independent); the similarities that depend on
                       lncRNA-disease links are recomputed here from the fold's training graph
"""
import os

import numpy as np
import pandas as pd

import config


def _dir(dataset):
    return f"./our_dataset/{dataset}"


def load_interactions(dataset):
    """lnc_di (L, D), lnc_mi (L, M), mi_di (M, D) association matrices."""
    read = lambda f: pd.read_csv(f"{_dir(dataset)}/interaction/{f}.csv", index_col="0").values
    return read("lnc_di"), read("lnc_mi"), read("mi_di")


def load_split(dataset, fold):
    """(positive_ij, positive5foldsidx, train positives, train U negatives, test positives, test U negatives)."""
    idx = f"{_dir(dataset)}/index"
    pf = np.load(f"{idx}/positive5foldsidx.npy", allow_pickle=True)
    nf = np.load(f"{idx}/negative5foldsidx.npy", allow_pickle=True)
    pos, neg = np.load(f"{idx}/positive_ij.npy"), np.load(f"{idx}/negative_ij.npy")
    return pos, pf, pos[pf[fold]["train"]], neg[nf[fold]["train"]], pos[pf[fold]["test"]], neg[nf[fold]["test"]]


def split_validation(train_pos, train_neg, dataset, fold):
    """(fit positives, fit negatives, validation positives, validation negatives).

    With index/val5foldsidx.npy (group-aware splits) the validation positives are the fold's held-out groups and the
    validation negatives every negative of a validation lncRNA (cold-start) or a random VAL_RATIO of them; otherwise a
    random VAL_RATIO of the positives and of the negatives.
    """
    seed = config.SEED + fold
    path = f"{_dir(dataset)}/index/val5foldsidx.npy"
    if os.path.exists(path):
        pre = np.load(path, allow_pickle=True)[fold]
        val_pairs = {tuple(p) for p in pre["pos_pairs"]}
        pm = np.array([tuple(p) in val_pairs for p in train_pos], dtype=bool)
        if pre["lnc"] is not None:
            nm = np.isin(train_neg[:, 0], pre["lnc"])
        else:
            nm = np.zeros(len(train_neg), dtype=bool)
            nm[np.random.default_rng(seed).permutation(len(train_neg))[:int(round(config.VAL_RATIO * len(train_neg)))]] = True
        return train_pos[~pm], train_neg[~nm], train_pos[pm], train_neg[nm]
    rng = np.random.default_rng(seed)
    parts = []
    for ij in (train_pos, train_neg):
        perm = rng.permutation(len(ij))
        n_val = int(round(config.VAL_RATIO * len(ij)))
        parts.append((ij[perm[n_val:]], ij[perm[:n_val]]))
    (pos_fit, pos_val), (neg_fit, neg_val) = parts
    return pos_fit, neg_fit, pos_val, neg_val


def degree_matched_negatives(dataset):
    """D negatives: one (l, d') per positive (l, d), d' ~ deg(d') among diseases not linked to l (index/negD_ij.npy
    when the split provides it)."""
    pre = f"{_dir(dataset)}/index/negD_ij.npy"
    if os.path.exists(pre):
        return np.load(pre)
    lnc_di = np.asarray(load_interactions(dataset)[0], dtype=float)
    n_lnc, n_dis = lnc_di.shape
    pos = np.load(f"{_dir(dataset)}/index/positive_ij.npy")
    deg_d, used, nonassoc = lnc_di.sum(0), lnc_di > 0, lnc_di == 0
    rng = np.random.default_rng(config.SEED)
    neg = np.empty_like(pos)
    for i, (l, _) in enumerate(pos):
        p = deg_d * ~used[l]
        if p.sum() == 0:
            p = (~used[l]).astype(float)
        if p.sum() == 0:
            p = deg_d * nonassoc[l]
        if p.sum() == 0:
            p = nonassoc[l].astype(float)
        d2 = rng.choice(n_dis, p=p / p.sum())
        used[l, d2] = True
        neg[i] = (l, d2 + n_lnc)
    return neg


def fold_sets(dataset, fold, negD_all):
    """{name: (positives, negatives)} of one fold.

    fit / val   training pairs, negatives = U + D (MIX training)
    U, D        test positives vs uniform / degree-matched negatives
    H, MIX      (splits with index/negH_fold<k>.npy) heuristic negatives; MIX = 1/3 U + 1/3 D + 1/3 H
    """
    _, pf, tr_pos, tr_negU, te_pos, te_negU = load_split(dataset, fold)
    fit_pos, fit_negU, val_pos, val_negU = split_validation(tr_pos, tr_negU, dataset, fold)
    fit_pos2, fit_negD, val_pos2, val_negD = split_validation(tr_pos, negD_all[pf[fold]["train"]], dataset, fold)
    assert np.array_equal(fit_pos, fit_pos2) and np.array_equal(val_pos, val_pos2)
    out = {"fit": (fit_pos, np.vstack([fit_negU, fit_negD])), "val": (val_pos, np.vstack([val_negU, val_negD])),
           "U": (te_pos, te_negU), "D": (te_pos, negD_all[pf[fold]["test"]])}
    h = f"{_dir(dataset)}/index/negH_fold{fold + 1}.npy"
    if os.path.exists(h):
        out["H"] = (te_pos, np.load(h)[:, :2])
        rng = np.random.default_rng(1000 + fold)
        n = len(te_pos)
        ks = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
        out["MIX"] = (te_pos, np.vstack([out[t][1][rng.choice(len(out[t][1]), k, replace=False)] for t, k in zip("UDH", ks)]))
    return out


# ------------------------------------------------------------------ similarities
def gip_similarity(matrix):
    """Gaussian interaction-profile kernels of the rows and of the columns (diagonal 0)."""
    def kernel(X):
        sq = (X ** 2).sum(axis=1)
        gamma = 1.0 / ((1.0 / X.shape[0]) * sq.sum())
        K = np.exp(-gamma * np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))
        np.fill_diagonal(K, 0)
        return K
    X = np.asarray(matrix, dtype=np.float64)
    return kernel(X), kernel(X.T)


def pbpa_similarity(di_sim, rna_di):
    """PBPA functional similarity of RNAs from their disease sets (reference loop of data_preprocessing.ipynb)."""
    n = rna_di.shape[0]
    sets = [rna_di[i] > 0 for i in range(n)]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            block = di_sim[sets[i]][:, sets[j]]
            if block.shape[0] == 0 or block.shape[1] == 0:
                continue
            S[i, j] = S[j, i] = (np.max(block, axis=0).sum() + np.max(block, axis=1).sum()) / (block.shape[0] + block.shape[1])
    return S


def fast_pbpa(sem, Y):
    """Vectorised PBPA (equal to pbpa_similarity up to ~1e-15), used for the many graphs of the stack."""
    n = Y.shape[0]
    Mx = np.zeros((n, sem.shape[1]))
    for i in range(n):
        t = np.flatnonzero(Y[i])
        if len(t):
            Mx[i] = sem[t].max(0)
    T = Mx @ Y.T
    cnt = Y.sum(1)
    den = cnt[:, None] + cnt[None, :]
    S = np.divide(T + T.T, den, out=np.zeros_like(T), where=(cnt[:, None] > 0) & (cnt[None, :] > 0))
    np.fill_diagonal(S, 0)
    return S


def fuse(F, G):
    """One similarity view per entity type: (F + G) symmetrised, divided by its maximum, diagonal 0."""
    M = np.asarray(F, dtype=np.float64) + np.asarray(G, dtype=np.float64)
    M = (M + M.T) / 2.0
    M = M / M.max()
    np.fill_diagonal(M, 0)
    return M


def mirna_jaccard(LM):
    """Jaccard similarity of the lncRNAs' miRNA sets (diagonal 0)."""
    B = (LM > 0).astype(float)
    inter = B @ B.T
    uni = B.sum(1)[:, None] + B.sum(1)[None, :] - inter
    J = np.divide(inter, uni, out=np.zeros_like(inter), where=uni > 0)
    np.fill_diagonal(J, 0)
    return J


# ------------------------------------------------------------------ training graph of a fold
def fold_inputs(dataset, fold):
    """Interaction matrices with the fold's test AND validation positives removed, and the 6 similarity views
    (lncRNA PBPA and GIP, disease semantic and GIP, miRNA functional and GIP); the views that depend on lncRNA-disease
    links are recomputed from the training graph."""
    lnc_di, lnc_mi, mi_di = (np.asarray(m, dtype=np.float64) for m in load_interactions(dataset))
    pos, pf, tr_pos, tr_neg, _, _ = load_split(dataset, fold)
    _, _, val_pos, _ = split_validation(tr_pos, tr_neg, dataset, fold)
    hidden = np.concatenate([pos[pf[fold]["test"]], val_pos])
    n_lnc = lnc_di.shape[0]
    lnc_di = lnc_di.copy()
    lnc_di[hidden[:, 0], hidden[:, 1] - n_lnc] = 0
    ms = f"{_dir(dataset)}/multi_similarities"
    di_semantic = np.load(f"{ms}/di_semantic_similarity.npy")
    lnc_gip, di_gip = gip_similarity(lnc_di)
    return {
        "lnc_di": lnc_di, "lnc_mi": lnc_mi, "mi_di": mi_di,
        "lnc_func": pbpa_similarity(di_semantic, lnc_di), "lnc_gip": lnc_gip,
        "di_semantic": di_semantic, "di_gip": di_gip,
        "mi_func": np.load(f"{ms}/mi_func_similarity.npy"), "mi_gip": np.load(f"{ms}/mi_gip_similarity.npy"),
    }


def assemble(lnc_sim, di_sim, mi_sim, lnc_di, lnc_mi, mi_di):
    """Heterogeneous adjacency [[LL, LD, LM], [DL, DD, DM], [ML, MD, MM]] over lncRNAs, diseases and miRNAs."""
    L, D, M = lnc_sim.shape[0], di_sim.shape[0], mi_sim.shape[0]
    A = np.zeros((L + D + M, L + D + M))
    A[:L, L:L + D] = lnc_di
    A[L:L + D, :L] = lnc_di.T
    A[:L, L + D:] = lnc_mi
    A[L + D:, :L] = lnc_mi.T
    A[L:L + D, L + D:] = mi_di.T
    A[L + D:, L:L + D] = mi_di
    A[:L, :L] = lnc_sim
    A[L:L + D, L:L + D] = di_sim
    A[L + D:, L + D:] = mi_sim
    return A
