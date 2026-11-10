"""LDAGM branch: masked VGAE node embeddings mu -> LDAGM pair classifier.

Graph of a fold: A = [[S_lnc, LD, LM], [DL, S_dis, DM], [ML, MD, S_mi]], with S = fused similarity view of each entity
type and the association blocks of the training graph (test and validation positives removed).
Encoder: VGAE-MLP on the rows of A (-> 256 -> mu / log_var of size 128, inner-product decoder). Every epoch a random
MASK_EDGE_RATE of the association edges, plus the lncRNA-disease rows of MASK_ROW_RATE of the lncRNAs, are removed
from the input; the loss reconstructs them, with the non-edges, on the association blocks only.
LDAGM: gated MLP (LDAGM.py) on [z(mu_l) || z(mu_d)], trained on the fit pairs (MIX negatives), epoch chosen on
validation AUC.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import config
import data
from LDAGM import LDAGM

CACHE = "./cache"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ masked VGAE
class VGAE(nn.Module):
    def __init__(self, n, hidden, latent):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(n, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.log_var = nn.Linear(hidden, latent)

    def forward(self, A):
        h = self.hidden(A)
        mu, log_var = self.mu(h), self.log_var(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var) if self.training else mu
        return z @ z.t(), mu, log_var


def association_mask(L, D, M):
    """Boolean mask of the lncRNA-disease, lncRNA-miRNA and miRNA-disease blocks (both directions)."""
    mask = np.zeros((L + D + M, L + D + M), dtype=bool)
    mask[:L, L:] = True
    mask[L:, :L] = True
    mask[L:L + D, L + D:] = True
    mask[L + D:, L:L + D] = True
    return mask


def masked_vgae_mu(A, dims, seed):
    """(N, EMBED_DIM) float32 mu of a masked VGAE trained on the adjacency A; dims = (n_lnc, n_dis, n_mi)."""
    L, D, M = dims
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    A = torch.as_tensor(A, dtype=torch.float32)
    mask = torch.as_tensor(association_mask(L, D, M))
    edges = torch.nonzero(torch.triu(mask, 1) & (A > 0.5))
    is_ld = (edges[:, 0] < L) & (edges[:, 1] >= L) & (edges[:, 1] < L + D)
    model = VGAE(A.shape[1], config.VGAE_HIDDEN_DIM, config.EMBED_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=config.VGAE_LR)
    n = A.shape[0]
    for _ in range(config.VGAE_EPOCHS):
        model.train()
        opt.zero_grad()
        drop = torch.rand(len(edges)) < config.MASK_EDGE_RATE
        rows = torch.as_tensor(rng.choice(L, size=max(1, int(config.MASK_ROW_RATE * L)), replace=False))
        drop |= is_ld & torch.isin(edges[:, 0], rows)
        h = edges[drop]
        A_in = A.clone()
        A_in[h[:, 0], h[:, 1]] = 0.0
        A_in[h[:, 1], h[:, 0]] = 0.0
        scored = mask & (A < 0.5)
        scored[h[:, 0], h[:, 1]] = True
        scored[h[:, 1], h[:, 0]] = True
        logits, mu, log_var = model(A_in)
        target = A[scored]
        n_pos = target.sum()
        recon = F.binary_cross_entropy_with_logits(logits[scored], target, pos_weight=(target.numel() - n_pos) / n_pos)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / n
        (recon + config.VGAE_KL_WEIGHT * kl).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        _, mu, _ = model(A)
    return mu.numpy().astype(np.float32)


# ------------------------------------------------------------------ LDAGM
class PairDataset(Dataset):
    def __init__(self, pos, neg):
        self.node_pairs = torch.tensor(np.vstack([pos, neg]), dtype=torch.long)
        self.labels = torch.tensor(np.hstack([np.ones(len(pos)), np.zeros(len(neg))]), dtype=torch.float32)

    def __getitem__(self, i):
        return self.node_pairs[i], self.labels[i]

    def __len__(self):
        return len(self.node_pairs)


def zscore(X):
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - X.mean(axis=0)) / sd


def pair_features(X, pairs):
    return torch.cat([X[pairs[:, 0]], X[pairs[:, 1]]], dim=1)


def score(model, X, ds):
    """(labels, sigmoid probabilities) of the pairs of ds."""
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(pair_features(X, ds.node_pairs))).numpy()
    return ds.labels.numpy(), p


def train_ldagm(X, train_ds, val_ds):
    """LDAGM on fixed node features, early stopping on validation AUC; returns (model, best_epoch, best_val_auc)."""
    model = LDAGM(input_dimension=2 * X.shape[1], hidden_dimension=config.LDAGM_HIDDEN_DIM, feature_num=1,
                  hiddenLayer_num=config.LDAGM_LAYERS, drop_rate=config.DROP_RATE, use_aggregate=config.USE_AGGREGATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    best_auc, best_epoch, best_state, waited = -np.inf, 0, None, 0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for pairs, labels in loader:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(pair_features(X, pairs)), labels)
            loss.backward()
            optimizer.step()
        val_auc = roc_auc_score(*score(model, X, val_ds))
        if val_auc > best_auc:
            best_auc, best_epoch, waited = val_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config.EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_auc


def ldagm_logits(dataset, fold, s, sets, x, keys, cache=CACHE):
    """{key: LDAGM logits of the pairs of sets[key]} for key in keys (cached per split, seed and fold).

    x = data.fold_inputs(dataset, fold); seed s: VGAE seed SEED + 100 fold + 1 + 10000 s, LDAGM seed SEED + fold + 10000 s.
    """
    path = f"{cache}/ldagm_{dataset}_s{s}_f{fold + 1}.npz"
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)["Lg"].item()
    n_lnc, n_dis = x["lnc_di"].shape
    dims = (n_lnc, n_dis, x["mi_di"].shape[0])
    sims = [data.fuse(x[a], x[b]) for a, b in (("lnc_func", "lnc_gip"), ("di_semantic", "di_gip"), ("mi_func", "mi_gip"))]
    mu = masked_vgae_mu(data.assemble(*sims, x["lnc_di"], x["lnc_mi"], x["mi_di"]), dims, config.SEED + 100 * fold + 1 + 10000 * s)
    ds = {k: PairDataset(*sets[k]) for k in ["fit"] + list(keys)}
    Xt = torch.tensor(zscore(mu).astype(np.float32))
    seed_all(config.SEED + fold + 10000 * s)
    model, _, _ = train_ldagm(Xt, ds["fit"], ds["val"])
    Lg = {k: logit(score(model, Xt, ds[k])[1]) for k in keys}
    os.makedirs(cache, exist_ok=True)
    np.savez(path, Lg=np.array(Lg, dtype=object))
    return Lg
