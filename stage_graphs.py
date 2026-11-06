"""Stage graphs A_1..A_K of one fold for the two-stage (fixed-embedding) pipeline.

From the stage-1 similarities of every entity type, F_1 (functional / semantic) and G_1 (GIP), stage k is a
personalized PageRank diffusion with restart probability alpha_k (closed form):
    M_1 = (F_1 + G_1) / max, diagonal 0;   P = D^-1 M_1
    F^(k) = alpha_k (I - (1 - alpha_k) P)^-1 F_1   (G^(k) likewise);   M_k = sym(F^(k) + G^(k)) / max, diagonal 0
and A_k = [[LM_k, LDM, LMM], [DLM, DM_k, DMM], [MLM, MDM, MM_k]] (Eq. 10 block layout).

The test AND validation positives of the fold (same validation split as fold_data_loader) are removed from the
lncRNA-disease matrix, and every similarity that depends on it (lncRNA functional similarity by PBPA,
lncRNA / disease GIP) is recomputed, so the encoders see neither the validation pairs used for early stopping nor
the test pairs. Nothing here has learnable parameters.
"""
import numpy as np

import config
import fold_data_loader as fdl


# ------------------------------------------------------------------ stage-1 similarities (notebook code)
def pbpa_similarity(di_sim, rna_di):
    """getRNASiNet / PBPA of data_preprocessing.ipynb (cells 9-10): functional similarity of RNAs."""
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


def gip_similarity(matrix):
    """getTuneParam + getGauSiNet of data_preprocessing.ipynb (cells 12-13), vectorised. Returns (rows, cols)."""
    def kernel(X):
        sq = (X ** 2).sum(axis=1)
        gamma = 1.0 / ((1.0 / X.shape[0]) * sq.sum())
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
        K = np.exp(-gamma * d2)
        np.fill_diagonal(K, 0)
        return K
    X = np.asarray(matrix, dtype=np.float64)
    return kernel(X), kernel(X.T)


def hidden_pairs(dataset, fold):
    """(i, j) pairs (j offset by n_lnc) hidden from the graphs: the test and the validation positives."""
    positive_ij, positive5foldsidx, train_pos, train_neg, _, _ = fdl._load_fold_split_indices(dataset, fold)
    _, _, val_pos, _ = fdl._split_validation(train_pos, train_neg, config.VAL_RATIO, config.SEED + fold)
    return np.concatenate([positive_ij[positive5foldsidx[fold]["test"]], val_pos])


def fold_training_inputs(dataset, fold):
    """Interaction matrices with the hidden positives removed, and the 6 stage-1 similarity views."""
    lnc_di, lnc_mi, mi_di = (np.asarray(m, dtype=np.float64) for m in fdl._load_interaction_matrices(dataset))
    n_lnc = lnc_di.shape[0]
    hidden = hidden_pairs(dataset, fold)
    lnc_di = lnc_di.copy()
    lnc_di[hidden[:, 0], hidden[:, 1] - n_lnc] = 0
    ms = f"./our_dataset/{dataset}/multi_similarities"
    di_semantic = np.load(f"{ms}/di_semantic_similarity.npy")          # does not use lncRNA-disease links
    lnc_gip, di_gip = gip_similarity(lnc_di)
    return {
        "lnc_di": lnc_di, "lnc_mi": lnc_mi, "mi_di": mi_di,
        "lnc_func": pbpa_similarity(di_semantic, lnc_di), "lnc_gip": lnc_gip,
        "di_semantic": di_semantic, "di_gip": di_gip,
        "mi_func": np.load(f"{ms}/mi_func_similarity.npy"), "mi_gip": np.load(f"{ms}/mi_gip_similarity.npy"),
    }


# ------------------------------------------------------------------ PPR stages
def ppr_stage_similarities(F1, G1, alphas):
    """Fused M_1..M_K, one PPR diffusion of the stage-1 similarities per restart probability alpha_k."""
    F1 = np.asarray(F1, dtype=np.float64)
    G1 = np.asarray(G1, dtype=np.float64)
    M1 = F1 + G1
    M1 = M1 / M1.max()
    np.fill_diagonal(M1, 0)
    row = M1.sum(axis=1, keepdims=True)
    P = np.divide(M1, row, out=np.zeros_like(M1), where=row > 0)
    eye = np.eye(F1.shape[0])
    fused = []
    for alpha in alphas:
        system = eye - (1.0 - alpha) * P
        Mk = alpha * np.linalg.solve(system, F1) + alpha * np.linalg.solve(system, G1)
        Mk = (Mk + Mk.T) / 2.0           # P is row-normalised, so the diffusion is not symmetric
        Mk = Mk / Mk.max()
        np.fill_diagonal(Mk, 0)
        fused.append(Mk)
    return fused


def assemble(lnc_sim, di_sim, mi_sim, lnc_di, lnc_mi, mi_di):
    """Heterogeneous adjacency (Eq. 10), same block layout as `cancatenate` in data_preprocessing.ipynb."""
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


def build_stage_graphs(dataset, fold, alphas):
    """A_1..A_K (float32, one per alpha_k) of one fold and the node counts (n_lnc, n_dis, n_mi)."""
    x = fold_training_inputs(dataset, fold)
    lnc = ppr_stage_similarities(x["lnc_func"], x["lnc_gip"], alphas)
    dis = ppr_stage_similarities(x["di_semantic"], x["di_gip"], alphas)
    mi = ppr_stage_similarities(x["mi_func"], x["mi_gip"], alphas)
    graphs = [assemble(lnc[k], dis[k], mi[k], x["lnc_di"], x["lnc_mi"], x["mi_di"]).astype(np.float32)
              for k in range(len(alphas))]
    return graphs, (x["lnc_di"].shape[0], x["lnc_di"].shape[1], x["mi_di"].shape[0])
