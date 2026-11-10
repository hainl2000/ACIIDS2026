"""Stack branch: 13 pair features of the training graph -> gradient boosting.

Features of a lncRNA-disease pair (l, d), all computed from a lncRNA-disease graph Y:
  KR func, KR GIP     KronRLS scores: lncRNA PBPA x disease semantic kernel, lncRNA GIP x disease GIP kernel
  log deg(l), (d)     log1p degrees
  log 3-path, RA      paths l-d'-l'-d (counts, resource allocation)
  log / RA miRNA      paths l-m-d through lncRNA-miRNA and miRNA-disease links
  shared-miRNA lnc    disease profile of the lncRNAs sharing miRNAs with l (Jaccard-weighted)
  max / mean sem      semantic similarity of d to l's known diseases
  GIP-CF lnc / dis    GIP-weighted collaborative filtering over lncRNAs / diseases
Training features are cross-fitted so that they look like test features: the fit pairs are cut into chunks the way
the test split cuts its positives, and the features of a chunk come from the graph without that chunk's positives.
"""
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

import config
import data

FEATURES = ["KR func", "KR GIP", "log deg(l)", "log deg(d)", "log 3-path", "RA 3-path", "log miRNA-path",
            "RA miRNA-path", "shared-miRNA lnc", "max sem", "mean sem", "GIP-CF lnc", "GIP-CF dis"]


# ------------------------------------------------------------------ KronRLS
def eig_psd(K):
    lam, U = np.linalg.eigh((K + K.T) / 2)
    return np.clip(lam, 0, None), U


def kronrls(el, ed, Y, sigma):
    (lam, U), (gam, V) = el, ed
    G = np.outer(lam, gam)
    return U @ ((G / (G + sigma)) * (U.T @ Y @ V)) @ V.T


def loo_kronrls(el, ed, Y, sigma):
    """Leave-one-out KronRLS scores (closed form), used to choose sigma on the validation pairs."""
    (lam, U), (gam, V) = el, ed
    G = np.outer(lam, gam)
    G = G / (G + sigma)
    Yhat = U @ (G * (U.T @ Y @ V)) @ V.T
    Hd = (U * U) @ G @ (V * V).T
    return (Yhat - Hd * Y) / np.clip(1.0 - Hd, 1e-6, None)


def kernels(Y, SEM):
    """(lncRNA PBPA, disease semantic) and (lncRNA GIP, disease GIP) kernel pairs of graph Y, diagonal 1."""
    lg, dg = data.gip_similarity(Y)
    kern = {"lnc_func": data.fast_pbpa(SEM, Y), "lnc_gip": lg, "di_semantic": SEM.copy(), "di_gip": dg}
    for k in kern:
        np.fill_diagonal(kern[k], 1.0)
    return [(kern["lnc_func"], kern["di_semantic"]), (kern["lnc_gip"], kern["di_gip"])]


def choose_sigmas(Y, SEM, val_pairs, y_val):
    """KronRLS sigma of each kernel pair by validation AUC of the leave-one-out scores on the fit graph."""
    n_lnc = Y.shape[0]
    sig = []
    for Kl, Kd in kernels(Y, SEM):
        el, ed = eig_psd(Kl), eig_psd(Kd)
        sig.append(max(config.KRONRLS_SIGMAS, key=lambda q: roc_auc_score(
            y_val, loo_kronrls(el, ed, Y, q)[val_pairs[:, 0], val_pairs[:, 1] - n_lnc])))
    return sig


# ------------------------------------------------------------------ feature maps
def _gip(X):
    sq = (X ** 2).sum(1)
    g = 1.0 / max(sq.mean(), 1e-12)
    K = np.exp(-g * np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    np.fill_diagonal(K, 0)
    return K


def path_maps(Y, LM, MD, SEM, LJ):
    """(L, D, 9) path / neighbourhood features of every lncRNA-disease pair of graph Y."""
    dl, dd = Y.sum(1), Y.sum(0)
    inv = lambda v: np.divide(1.0, v, out=np.zeros_like(v, dtype=float), where=v > 0)
    P3 = Y @ Y.T @ Y
    RA3 = (Y * inv(dd)[None, :]) @ (Y.T * inv(dl)[None, :]) @ Y
    dm = LM.sum(0) + MD.sum(1)
    M2 = LM @ MD
    RA2 = (LM * inv(dm)[None, :]) @ MD
    MIL = (LJ @ Y) * inv(LJ.sum(1))[:, None]
    maxsem = np.zeros_like(Y, dtype=float)
    for l in range(Y.shape[0]):
        t = np.flatnonzero(Y[l])
        if len(t):
            maxsem[l] = SEM[:, t].max(1)
    meansem = (Y @ SEM.T) * inv(dl)[:, None]
    GL, GD = _gip(Y), _gip(Y.T)
    cf_l = (GL @ Y) * inv(GL.sum(1))[:, None]
    cf_d = (Y @ GD) * inv(GD.sum(0))[None, :]
    return np.stack([np.log1p(P3), RA3, np.log1p(M2), RA2, MIL, maxsem, meansem, cf_l, cf_d], -1)


def feature_maps(Y, SEM, LM, MD, LJ, sig):
    """(L, D, 13) stack features of every pair, from graph Y (kernels recomputed from Y)."""
    kr = [kronrls(eig_psd(Kl), eig_psd(Kd), Y, s) for (Kl, Kd), s in zip(kernels(Y, SEM), sig)]
    dl = np.broadcast_to(np.log1p(Y.sum(1, keepdims=True)), Y.shape)
    dd = np.broadcast_to(np.log1p(Y.sum(0, keepdims=True)), Y.shape)
    return np.concatenate([np.stack(kr + [dl, dd], -1), path_maps(Y, LM, MD, SEM, LJ)], -1)


# ------------------------------------------------------------------ split-consistent cross-fitting
def family_groups(Y, cols, G):
    """Per lncRNA, its diseases joined by direct Disease-Ontology is_a edges (as build_eval_splits.py)."""
    groups = []
    for l in range(Y.shape[0]):
        ds = np.flatnonzero(Y[l])
        H = nx.Graph()
        H.add_nodes_from(ds)
        name = {cols[d]: d for d in ds}
        for nm, d in name.items():
            if nm in G:
                for parent in G.successors(nm):
                    if parent in name:
                        H.add_edge(d, name[parent])
        groups += [[(l, d) for d in c] for c in nx.connected_components(H)]
    return groups


def consistent_chunks(dataset, Y, fit_pairs, fit_y, cols, G, A, rng):
    """[(Y_c, mask over the fit pairs)] for CROSS_FIT_CHUNKS chunks cut like the split's test positives:
      _cs    lncRNA chunks (a chunk lncRNA loses every link)
      _fam   (lncRNA, disease family) groups
      _leaf  leaf chunks: a chunk removes its leaves and every ancestor no other leaf supports (A = DO ancestor matrix);
             positives never removed are not used, negatives below a remaining leaf are dropped (as the test)
      else   pair chunks
    Fit pairs outside every mask are not used to fit the stack."""
    K = config.CROSS_FIT_CHUNKS
    n_lnc = Y.shape[0]
    l, d = fit_pairs[:, 0], fit_pairs[:, 1] - n_lnc
    out = []
    if dataset.endswith("_cs"):
        lnc = np.unique(l)
        ch = dict(zip(lnc, rng.integers(0, K, len(lnc))))
        cl = np.array([ch[x] for x in l])
        for c in range(K):
            Yc = Y.copy()
            Yc[np.isin(np.arange(n_lnc), [x for x, v in ch.items() if v == c])] = 0
            out.append((Yc, cl == c))
    elif dataset.endswith("_fam"):
        gid = -np.ones(Y.shape, dtype=int)
        groups = family_groups(Y, cols, G)
        gch = rng.integers(0, K, len(groups))
        for g, members in enumerate(groups):
            for (a, b) in members:
                gid[a, b] = gch[g]
        cpair = np.where(fit_y == 1, gid[l, d], rng.integers(0, K, len(l)))
        for c in range(K):
            Yc = Y.copy()
            Yc[gid == c] = 0
            out.append((Yc, cpair == c))
    elif dataset.endswith("_leaf"):
        leaves = np.argwhere((Y > 0) & ((Y @ A) == 0))
        lch = rng.integers(0, K, len(leaves))

        def ind(Lv):
            M = np.zeros(Y.shape)
            M[Lv[:, 0], Lv[:, 1]] = 1.0
            return M
        neg_ch = rng.integers(0, K, len(l))
        for c in range(K):
            keep = leaves[lch != c]
            Yc = np.maximum(ind(keep), (ind(keep) @ A) > 0).astype(float)
            removed = (Y > 0) & (Yc == 0)
            free = (ind(keep) @ A.T) > 0
            m = np.where(fit_y == 1, removed[l, d], (neg_ch == c) & ~free[l, d])
            out.append((Yc, m))
    else:
        cp = rng.integers(0, K, len(l))
        for c in range(K):
            Yc = Y.copy()
            sel = (cp == c) & (fit_y == 1)
            Yc[l[sel], d[sel]] = 0
            out.append((Yc, cp == c))
    return out


def cross_fitted_features(dataset, Y, SEM, LM, MD, LJ, sig, fit_pairs, fit_y, cols, G, A, rng):
    """(features of the fit pairs, mask of the fit pairs that have them)."""
    n_lnc = Y.shape[0]
    X = np.full((len(fit_pairs), len(FEATURES)), np.nan)
    for Yc, m in consistent_chunks(dataset, Y, fit_pairs, fit_y, cols, G, A, rng):
        if m.any():
            Fc = feature_maps(Yc, SEM, LM, MD, LJ, sig)
            X[m] = Fc[fit_pairs[m, 0], fit_pairs[m, 1] - n_lnc]
    return X, ~np.isnan(X[:, 0])
