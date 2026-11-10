"""Leakage-aware evaluation splits with mixed negatives for the v2026 datasets.

Positive splits (5 folds, groups never cross folds)
  dataset1_v2026 -> dataset1_v2026_cs  : cold-start lncRNA. Every association of a test lncRNA is a test pair (the
                                         Fu-type labels are propagated to ancestor diseases, so all associations of one
                                         lncRNA form a single disease family; Park & Marcotte 2012 class C2).
  dataset2_v2026 -> dataset2_v2026_fam : disease-family split. For each lncRNA, its associated diseases joined by direct
                                         Disease Ontology is_a edges form a family; a family stays in one fold.
  dataset2_v2026 -> dataset2_v2026_cs  : cold-start lncRNA on the Zhou benchmark.
  dataset1_v2026 -> dataset1_v2026_fam : disease-family split on Fu (close to its cold-start split, see JOBS).
  dataset1_v2026 -> dataset1_v2026_leaf: leaf split ("P2x") that removes the Disease-Ontology hierarchy leak of the
                                         propagated Fu labels. Labels are first completed by the true-path rule (the
                                         few unlabelled DO ancestors of a positive become positives). The most specific
                                         positives of each lncRNA ("leaves") are split into folds; a fold's training
                                         positives are its training leaves plus all their DO ancestors, its test
                                         positives everything not derivable from them (so no test positive is implied
                                         by a training one, and no training label was derived from a test one). Test
                                         negatives (U, D, H) exclude the DO descendants of the training leaves of the
                                         same lncRNA: such pairs can never be positive in the data, so they would be
                                         free negatives for a hierarchy-aware model. Test positives do not partition
                                         the positives: an ancestor supported by leaves of two folds is always training.
Negatives (each 1:1 with the positives of a split; written to index/)
  U  uniform     : unknown pairs drawn at random (negative_ij + negative5foldsidx, read by every pipeline).
                   For the cold-start split train / test negatives use train / test lncRNAs only.
  D  degree      : one (l, d') per positive (l, d), d' ~ deg(d') among diseases not linked to l (negD_ij.npy, aligned
                   with positive_ij, so a D negative inherits the fold of its positive).
  H  heuristic   : test only. One (l, d') per test positive, d' drawn from the top 10% of l's non-associated diseases
                   under one of three training-graph heuristics, used in turn (HeaRT-style, Li et al. 2023):
                   3-path l-d'-l'-d (Y Y^T Y), miRNA path l-m-d (LM MD), semantic similarity of d to l's training
                   diseases. If none is informative for l (e.g. a cold-start lncRNA without miRNA links), d' falls back
                   to the degree rule and the pair is flagged. index/negH_fold<k>.npy, columns l, d+n_lnc, heuristic id.
Only index/ differs from the source dataset; basic/ and interaction/ are copied unchanged.

Usage: python build_eval_splits.py [destination ...] [--overwrite]   (existing destinations are skipped)
"""
import json
import os
import shutil
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import obonet
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
DOID_OBO = os.path.join(os.path.dirname(REPO), "NKHTrẻ2026", "Dataset", "doid.obo")
SEED = 42
K = 5
VAL_RATIO = 0.1                # validation = whole groups holding ~10% of each fold's training positives
TOP = 0.10
HEURISTICS = ("3-path", "miRNA-path", "semantic")
JOBS = {"dataset1_v2026_cs": ("dataset1_v2026", "cold-start lncRNA"),     # destination -> (source, split)
        "dataset2_v2026_fam": ("dataset2_v2026", "disease family"),
        "dataset2_v2026_cs": ("dataset2_v2026", "cold-start lncRNA"),
        # Fu-type labels put ~98% of a lncRNA's links in one family, so this is close to the cold-start split
        "dataset1_v2026_fam": ("dataset1_v2026", "disease family"),
        "dataset1_v2026_leaf": ("dataset1_v2026", "leaf (P2x)")}


def load(src):
    read = lambda f: pd.read_csv(f"{REPO}/our_dataset/{src}/interaction/{f}.csv", index_col="0")
    ld, lm, md = read("lnc_di"), read("lnc_mi"), read("mi_di")
    sem = np.load(f"{REPO}/our_dataset/{src}/multi_similarities/di_semantic_similarity.npy")
    return ld, lm.to_numpy(float), md.to_numpy(float), sem


def groups_for(ld, mode, G):
    """List of groups, each a list of positive indices (positives in row-major order, as the preprocessing notebook)."""
    Y = ld.to_numpy()
    pos = np.argwhere(Y == 1)
    by_lnc = defaultdict(list)
    for k, (l, d) in enumerate(pos):
        by_lnc[l].append(k)
    if mode == "cold-start lncRNA":
        return list(by_lnc.values()), pos
    cols = list(ld.columns)
    groups = []
    for l, idx in by_lnc.items():
        H = nx.Graph()
        H.add_nodes_from(idx)
        dis = {cols[pos[k][1]]: k for k in idx}
        for name, k in dis.items():
            if name in G:
                for parent in G.successors(name):
                    if parent in dis:
                        H.add_edge(k, dis[parent])
        groups += [sorted(c) for c in nx.connected_components(H)]
    return groups, pos


def assign_folds(groups, rng):
    """Greedy size balancing: largest group first, into the fold with the fewest positives (random tie-break)."""
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), rng.random()))
    size, fold_of = np.zeros(K), {}
    for g in order:
        f = int(np.flatnonzero(size == size.min())[rng.integers((size == size.min()).sum())])
        fold_of[g] = f
        size[f] += len(groups[g])
    folds = [[] for _ in range(K)]
    for g, f in fold_of.items():
        folds[f] += groups[g]
    return [np.array(sorted(x)) for x in folds]


def degree_negatives(Y, pos, rng):
    deg_d, used, nonassoc = Y.sum(0), Y > 0, Y == 0
    neg = np.empty_like(pos)
    for i, (l, _) in enumerate(pos):
        for p in (deg_d * ~used[l], (~used[l]).astype(float), deg_d * nonassoc[l], nonassoc[l].astype(float)):
            if p.sum() > 0:
                break
        d2 = rng.choice(Y.shape[1], p=p / p.sum())
        used[l, d2] = True
        neg[i] = (l, d2)
    return neg


def heuristic_negatives(Y, Ytr, LM, MD, sem, test_pos, rng, allow=None):
    """One hard negative per test positive; returns (n, 3) array: l, d, heuristic id (3 = degree fallback).
    allow (optional, lncRNA x disease bool): pairs that may be drawn (the leaf split excludes free negatives)."""
    S = [(Ytr @ Ytr.T @ Ytr).astype(float), LM @ MD, np.zeros(Ytr.shape, dtype=float)]
    for l in range(Y.shape[0]):
        tr = np.flatnonzero(Ytr[l])
        if len(tr):
            S[2][l] = sem[:, tr].max(1)
    deg_d, used = Y.sum(0), Y > 0
    out = []
    for i, (l, _) in enumerate(test_pos):
        ok_l = np.ones(Y.shape[1], dtype=bool) if allow is None else allow[l]
        cand = ~used[l] & ok_l
        if not cand.any():                    # lncRNA linked to > half of the diseases: repeats unavoidable
            cand = (Y[l] == 0) & ok_l
        pick = None
        for t in range(3):
            h = (i + t) % 3
            sc = np.where(cand, S[h][l], 0.0)
            ok = np.flatnonzero(sc > 0)
            if len(ok):
                top = ok[np.argsort(-sc[ok])][:max(1, int(np.ceil(TOP * cand.sum())))]
                pick = (rng.choice(top), h)
                break
        if pick is None:
            p = deg_d * cand if (deg_d * cand).sum() > 0 else cand.astype(float)
            pick = (rng.choice(Y.shape[1], p=p / p.sum()), 3)
        used[l, pick[0]] = True
        out.append((l, pick[0], pick[1]))
    return np.array(out)


def uniform_negatives(Y, folds_pos, pos, cold_start, rng):
    neg = np.argwhere(Y == 0)
    nidx = []
    if cold_start:
        lnc_fold = {}
        for f, idx in enumerate(folds_pos):
            for k in idx:
                lnc_fold[pos[k][0]] = f
        fold_of_neg = np.array([lnc_fold.get(l, -1) for l in neg[:, 0]])
        for f, idx in enumerate(folds_pos):
            n_te, n_tr = len(idx), sum(len(x) for g, x in enumerate(folds_pos) if g != f)
            te_pool = np.flatnonzero(fold_of_neg == f)
            tr_pool = np.flatnonzero((fold_of_neg != f) & (fold_of_neg >= 0))
            nidx.append({"train": rng.choice(tr_pool, n_tr, replace=False), "test": rng.choice(te_pool, n_te, replace=False)})
    else:
        perm = rng.permutation(len(neg))[:len(pos)]
        start, chunks = 0, []
        for idx in folds_pos:
            chunks.append(perm[start:start + len(idx)])
            start += len(idx)
        for f in range(K):
            nidx.append({"train": np.concatenate([chunks[g] for g in range(K) if g != f]), "test": chunks[f]})
    return neg, nidx


def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def build(src, dst, mode, G):
    rng = np.random.default_rng(SEED)
    ld, LM, MD, sem = load(src)
    Y = ld.to_numpy()
    n_lnc = Y.shape[0]
    groups, pos = groups_for(ld, mode, G)
    folds_pos = assign_folds(groups, rng)
    cold = mode == "cold-start lncRNA"
    neg, nidx = uniform_negatives(Y, folds_pos, pos, cold, rng)
    negD = degree_negatives(Y, pos, rng)

    out = f"{REPO}/our_dataset/{dst}"
    if os.path.exists(out):
        shutil.rmtree(out)
    for sub in ("basic", "interaction"):
        shutil.copytree(f"{REPO}/our_dataset/{src}/{sub}", f"{out}/{sub}")
    os.makedirs(f"{out}/index")
    allidx = np.arange(len(pos))
    pf = np.array([{"train": np.setdiff1d(allidx, folds_pos[f]), "test": folds_pos[f]} for f in range(K)])
    nf = np.array(nidx)
    off = lambda a: np.column_stack([a[:, 0], a[:, 1] + n_lnc])
    np.save(f"{out}/index/positive_ij.npy", off(pos))
    np.save(f"{out}/index/negative_ij.npy", off(neg))
    np.save(f"{out}/index/positive5foldsidx.npy", pf, allow_pickle=True)
    np.save(f"{out}/index/negative5foldsidx.npy", nf, allow_pickle=True)
    np.save(f"{out}/index/negD_ij.npy", off(negD))
    # group-aware validation split (own RNG, so the test split and the negatives above are unchanged);
    # read by fold_data_loader._split_validation
    val = []
    for f in range(K):
        vr = np.random.default_rng(SEED + 1000 + f)
        test = set(folds_pos[f].tolist())
        cand = [g for g in groups if g[0] not in test]
        target, chosen, count = VAL_RATIO * (len(pos) - len(test)), [], 0
        for k in vr.permutation(len(cand)):
            if count >= target:
                break
            chosen += cand[k]
            count += len(cand[k])
        chosen = np.array(sorted(chosen))
        val.append({"pos_pairs": off(pos[chosen]), "lnc": np.unique(pos[chosen, 0]) if cold else None})
    np.save(f"{out}/index/val5foldsidx.npy", np.array(val, dtype=object), allow_pickle=True)

    cols = list(ld.columns)
    report = {"source": src, "split": mode, "folds": []}
    lines = [f"{src} -> {dst} ({mode}): {len(pos)} positives in {len(groups)} groups"]
    deg_rule = lambda Ytr, pairs: np.log1p(Ytr.sum(1))[pairs[:, 0]] + np.log1p(Ytr.sum(0))[pairs[:, 1]]
    for f in range(K):
        te, tr = pf[f]["test"], pf[f]["train"]
        Ytr = np.zeros_like(Y)
        Ytr[pos[tr, 0], pos[tr, 1]] = 1
        negH = heuristic_negatives(Y, Ytr, LM, MD, sem, pos[te], rng)
        np.save(f"{out}/index/negH_fold{f + 1}.npy", np.column_stack([negH[:, 0], negH[:, 1] + n_lnc, negH[:, 2]]))
        # checks
        te_l = set(pos[te, 0])
        if cold:
            assert not (te_l & set(pos[tr, 0])), "a test lncRNA has training positives"
            assert not (te_l & set(neg[nf[f]["train"], 0])), "a test lncRNA has training negatives"
        else:
            direct = 0
            vis = defaultdict(set)
            for l, d in pos[tr]:
                vis[l].add(cols[d])
            for l, d in pos[te]:
                nm = cols[d]
                rel = (set(G.successors(nm)) | set(G.predecessors(nm))) if nm in G else set()
                direct += bool(rel & vis[l])
            assert direct == 0, f"{direct} test positives have a direct parent/child of the same lncRNA in training"
        for arr in (neg[nf[f]["test"]], negD[te], negH[:, :2]):
            assert Y[arr[:, 0], arr[:, 1]].sum() == 0, "a negative is a known association"
        # trivial baselines per negative type (positives vs that type, 1:1)
        S1 = Ytr @ Ytr.T @ Ytr
        S2 = LM @ MD
        S3 = np.zeros_like(Ytr, dtype=float)
        for l in range(n_lnc):
            t = np.flatnonzero(Ytr[l])
            if len(t):
                S3[l] = sem[:, t].max(1)
        rules = {"degree": lambda p: deg_rule(Ytr, p), "3-path": lambda p: S1[p[:, 0], p[:, 1]],
                 "miRNA-path": lambda p: S2[p[:, 0], p[:, 1]], "semantic": lambda p: S3[p[:, 0], p[:, 1]]}
        row = {"fold": f + 1, "test_pos": int(len(te)), "train_pos": int(len(tr)),
               "H_heuristic_share": {HEURISTICS[h] if h < 3 else "degree fallback": float((negH[:, 2] == h).mean()) for h in range(4)}}
        for name, negs in (("U", neg[nf[f]["test"]]), ("D", negD[te]), ("H", negH[:, :2])):
            pairs = np.vstack([pos[te], negs])
            y = np.r_[np.ones(len(te)), np.zeros(len(negs))]
            row[f"rule_auc_{name}"] = {r: float(auc(y, fn(pairs))) for r, fn in rules.items()}
        report["folds"].append(row)
    with open(f"{out}/index/split_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    sizes = [r["test_pos"] for r in report["folds"]]
    lines.append(f"  test positives per fold {sizes}; checks passed (no group crosses folds, negatives are unknown pairs)")
    share = np.mean([[r["H_heuristic_share"][k] for k in r["H_heuristic_share"]] for r in report["folds"]], 0)
    lines.append("  H negatives by heuristic: " + ", ".join(f"{k} {v:.0%}" for k, v in zip(list(report["folds"][0]["H_heuristic_share"]), share)))
    lines.append(f"  {'rule AUC (no learning)':24s} {'test-U':>8s} {'test-D':>8s} {'test-H':>8s}")
    for r in ("degree", "3-path", "miRNA-path", "semantic"):
        lines.append(f"  {r:24s} " + " ".join(f"{np.mean([x[f'rule_auc_{t}'][r] for x in report['folds']]):8.4f}" for t in "UDH"))
    return lines


def ancestor_matrix(cols, G):
    """A[j, k] = 1 when disease k is a Disease-Ontology (is_a) ancestor of disease j; dataset diseases only."""
    Gi = nx.DiGraph([(u, v) for u, v, k in G.edges(keys=True) if k == "is_a"])      # child -> parent
    idx = {d: j for j, d in enumerate(cols)}
    A = np.zeros((len(cols), len(cols)))
    for j, d in enumerate(cols):
        for a in (nx.descendants(Gi, d) if d in Gi else ()):
            if a in idx:
                A[j, idx[a]] = 1.0
    return A


def build_leaf(src, dst, G):
    """Leaf split (P2x) of a dataset whose labels follow the true-path rule; see the module docstring."""
    rng = np.random.default_rng(SEED)
    ld, LM, MD, sem = load(src)
    cols = list(ld.columns)
    Y0 = ld.to_numpy().astype(int)
    n_lnc, n_dis = Y0.shape
    A = ancestor_matrix(cols, G)
    implicit = (Y0 == 0) & ((Y0 @ A) > 0)                 # unlabelled DO ancestors of a positive
    Y = np.maximum(Y0, implicit).astype(int)
    pos = np.argwhere(Y == 1)                              # row-major, as the preprocessing notebook
    posidx = -np.ones(Y.shape, dtype=int)
    posidx[pos[:, 0], pos[:, 1]] = np.arange(len(pos))
    leaves = np.argwhere((Y == 1) & ((Y @ A) == 0))
    leaf_fold = np.empty(len(leaves), dtype=int)
    leaf_fold[rng.permutation(len(leaves))] = np.arange(len(leaves)) % K

    def ind(L):
        M = np.zeros(Y.shape)
        M[L[:, 0], L[:, 1]] = 1.0
        return M
    closure = lambda L: np.maximum(ind(L), (ind(L) @ A) > 0)          # leaves + all their DO ancestors
    off = lambda a: np.column_stack([a[:, 0], a[:, 1] + n_lnc])

    pf, val, free, Ytrs = [], [], [], []
    for f in range(K):
        tr_l = np.flatnonzero(leaf_fold != f)
        Ytr = closure(leaves[tr_l])
        pf.append({"train": np.sort(posidx[(Y == 1) & (Ytr > 0)]), "test": np.sort(posidx[(Y == 1) & (Ytr == 0)])})
        free.append((ind(leaves[tr_l]) @ A.T) > 0)        # below a training leaf of the same lncRNA
        Ytrs.append(Ytr)
        vr = np.random.default_rng(SEED + 1000 + f)        # validation = whole training leaves (~10%)
        v_l = vr.permutation(tr_l)[:int(round(VAL_RATIO * len(tr_l)))]
        Yfit = closure(leaves[np.setdiff1d(tr_l, v_l)])
        val.append({"pos_pairs": off(np.argwhere((Ytr > 0) & (Yfit == 0))), "lnc": None})
    test_fold = -np.ones(len(pos), dtype=int)
    for f in range(K):
        test_fold[pf[f]["test"]] = f

    neg = np.argwhere(Y == 0)                              # U: test negatives avoid the fold's free pairs
    nidx = []
    for f in range(K):
        ok = ~free[f][neg[:, 0], neg[:, 1]]
        te_n = rng.choice(np.flatnonzero(ok), len(pf[f]["test"]), replace=False)
        rest = np.setdiff1d(np.arange(len(neg)), te_n)
        nidx.append({"train": np.sort(rng.choice(rest, len(pf[f]["train"]), replace=False)), "test": np.sort(te_n)})
    deg_d, used = Y.sum(0), Y > 0                          # D: one per positive, inherits its test fold's exclusion
    negD = np.empty_like(pos)
    for k, (l, _) in enumerate(pos):
        allow = ~free[test_fold[k]][l] if test_fold[k] >= 0 else np.ones(n_dis, dtype=bool)
        for p in (deg_d * ~used[l] * allow, (~used[l] & allow).astype(float), deg_d * (Y[l] == 0) * allow,
                  ((Y[l] == 0) & allow).astype(float)):
            if p.sum() > 0:
                break
        d2 = rng.choice(n_dis, p=p / p.sum())
        used[l, d2] = True
        negD[k] = (l, d2)

    out = f"{REPO}/our_dataset/{dst}"
    if os.path.exists(out):
        shutil.rmtree(out)
    for sub in ("basic", "interaction"):
        shutil.copytree(f"{REPO}/our_dataset/{src}/{sub}", f"{out}/{sub}")
    pd.DataFrame(Y, index=ld.index, columns=ld.columns).to_csv(f"{out}/interaction/lnc_di.csv")   # completed labels
    os.makedirs(f"{out}/multi_similarities")
    for fn in ("di_semantic_similarity", "mi_func_similarity", "mi_gip_similarity"):              # fold-independent
        shutil.copy(f"{REPO}/our_dataset/{src}/multi_similarities/{fn}.npy", f"{out}/multi_similarities/{fn}.npy")
    os.makedirs(f"{out}/index")
    np.save(f"{out}/index/positive_ij.npy", off(pos))
    np.save(f"{out}/index/negative_ij.npy", off(neg))
    np.save(f"{out}/index/positive5foldsidx.npy", np.array(pf), allow_pickle=True)
    np.save(f"{out}/index/negative5foldsidx.npy", np.array(nidx), allow_pickle=True)
    np.save(f"{out}/index/negD_ij.npy", off(negD))
    np.save(f"{out}/index/val5foldsidx.npy", np.array(val, dtype=object), allow_pickle=True)
    np.save(f"{out}/index/leaves.npy", np.column_stack([off(leaves), leaf_fold]))
    np.save(f"{out}/index/implicit_added.npy", off(np.argwhere(implicit)))

    report = {"source": src, "split": "leaf (P2x)", "implicit_added": int(implicit.sum()), "leaves": int(len(leaves)),
              "folds": []}
    lines = [f"{src} -> {dst} (leaf split P2x): {int(Y0.sum())} labelled + {int(implicit.sum())} unlabelled DO ancestors "
             f"= {len(pos)} positives; {len(leaves)} leaves"]
    S2 = LM @ MD
    for f in range(K):
        te, tr, Ytr = pf[f]["test"], pf[f]["train"], Ytrs[f]
        negH = heuristic_negatives(Y, Ytr.astype(int), LM, MD, sem, pos[te], rng, allow=~free[f])
        np.save(f"{out}/index/negH_fold{f + 1}.npy", np.column_stack([negH[:, 0], negH[:, 1] + n_lnc, negH[:, 2]]))
        # checks
        assert ((Ytr @ A) > 0)[pos[te, 0], pos[te, 1]].sum() == 0, "a test positive is implied by a training positive"
        assert Ytr[pos[te, 0], pos[te, 1]].sum() == 0 and Ytr[pos[tr, 0], pos[tr, 1]].min() == 1
        for arr in (neg[nidx[f]["test"]], negD[te], negH[:, :2]):
            assert Y[arr[:, 0], arr[:, 1]].sum() == 0, "a negative is a known association"
            assert free[f][arr[:, 0], arr[:, 1]].sum() == 0, "a test negative lies below a training leaf"
        S1 = Ytr @ Ytr.T @ Ytr
        S3 = np.zeros_like(Ytr, dtype=float)
        for l in range(n_lnc):
            t = np.flatnonzero(Ytr[l])
            if len(t):
                S3[l] = sem[:, t].max(1)
        dr = lambda p: np.log1p(Ytr.sum(1))[p[:, 0]] + np.log1p(Ytr.sum(0))[p[:, 1]]
        rules = {"degree": dr, "3-path": lambda p: S1[p[:, 0], p[:, 1]], "miRNA-path": lambda p: S2[p[:, 0], p[:, 1]],
                 "semantic": lambda p: S3[p[:, 0], p[:, 1]]}
        is_leaf = np.zeros(Y.shape, dtype=bool)
        is_leaf[leaves[:, 0], leaves[:, 1]] = True
        row = {"fold": f + 1, "test_pos": int(len(te)), "test_leaves": int(is_leaf[pos[te, 0], pos[te, 1]].sum()),
               "train_pos": int(len(tr)), "free_share": float(free[f][Y == 0].mean()),
               "H_heuristic_share": {HEURISTICS[h] if h < 3 else "degree fallback": float((negH[:, 2] == h).mean()) for h in range(4)}}
        for name, negs in (("U", neg[nidx[f]["test"]]), ("D", negD[te]), ("H", negH[:, :2])):
            pairs = np.vstack([pos[te], negs])
            y = np.r_[np.ones(len(te)), np.zeros(len(negs))]
            row[f"rule_auc_{name}"] = {r: float(auc(y, fn(pairs))) for r, fn in rules.items()}
        report["folds"].append(row)
    with open(f"{out}/index/split_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    fr = report["folds"]
    lines.append(f"  test positives per fold {[r['test_pos'] for r in fr]} (leaves {[r['test_leaves'] for r in fr]}); "
                 f"training positives per fold {[r['train_pos'] for r in fr]}")
    lines.append(f"  non-positive pairs below a training leaf (excluded from test negatives): "
                 f"{np.mean([r['free_share'] for r in fr]):.1%}; checks passed (no test positive implied by training, "
                 f"negatives unknown and not below a training leaf)")
    share = np.mean([[r["H_heuristic_share"][k] for k in r["H_heuristic_share"]] for r in fr], 0)
    lines.append("  H negatives by heuristic: " + ", ".join(f"{k} {v:.0%}" for k, v in zip(list(fr[0]["H_heuristic_share"]), share)))
    lines.append(f"  {'rule AUC (no learning)':24s} {'test-U':>8s} {'test-D':>8s} {'test-H':>8s}")
    for r in ("degree", "3-path", "miRNA-path", "semantic"):
        lines.append(f"  {r:24s} " + " ".join(f"{np.mean([x[f'rule_auc_{t}'][r] for x in fr]):8.4f}" for t in "UDH"))
    return lines


if __name__ == "__main__":
    # build() deletes the destination folder, including the A/, offline/ ... written later by the pipeline,
    # so an existing split is rebuilt only with --overwrite
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(JOBS)
    G = obonet.read_obo(DOID_OBO)
    for dst in names:
        src, mode = JOBS[dst]
        if os.path.exists(f"{REPO}/our_dataset/{dst}") and "--overwrite" not in sys.argv:
            print(f"{dst} exists, skipped (use --overwrite to rebuild it)", flush=True)
            continue
        print("\n".join(build_leaf(src, dst, G) if mode == "leaf (P2x)" else build(src, dst, mode, G)), flush=True)
