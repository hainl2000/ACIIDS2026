"""Offline encoders of the two-stage pipeline. Both are trained on a FIXED graph A_k with a FIXED target.

GCN-AE   : the model of obtain_nonlinear_features.ipynb (cell 3), same training settings (MSE on A_k).
VGAE-MLP : variational graph auto-encoder whose encoder is an MLP on the rows of A_k (no GCN propagation;
           the multi-step propagation is already in the PPR graph). Inner-product decoder; the reconstruction
           target is the association blocks of A_k only (lncRNA-disease, lncRNA-miRNA, miRNA-disease),
           so the dense similarity blocks do not dominate the loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------------------------
# GCN-AE, copied from obtain_nonlinear_features.ipynb (cell 3)
# ----------------------------------------------------------------------------------------------
class LplsNorm(nn.Module):
    def forward(self, adjacency_matrix):
        degree = torch.sum(adjacency_matrix, dim=-1)
        diag = torch.diag(torch.pow(degree, -0.5))
        return diag.mm(adjacency_matrix).mm(diag)


class Convolution(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super().__init__()
        self.LinearLayer = nn.Sequential(nn.Linear(in_dimension, out_dimension), nn.ReLU())

    def forward(self, adjacency_matrix, feature):
        return self.LinearLayer(feature + torch.mm(adjacency_matrix, feature))


class ConvolutionLayer(nn.Module):
    def __init__(self, in_dimension, embedding_dimension):
        super().__init__()
        self.LplsNorm = LplsNorm()
        self.Convolution = Convolution(in_dimension, embedding_dimension)

    def forward(self, adjacency_matrix, feature):
        return self.Convolution(self.LplsNorm(adjacency_matrix), feature)


class GraphConvolution(nn.Module):
    def __init__(self, in_dimension, embedding_dimension):
        super().__init__()
        self.ConvolutionLayer = ConvolutionLayer(in_dimension, embedding_dimension)
        self.LinearLayer = nn.Sequential(
            nn.Linear(embedding_dimension, in_dimension), nn.ReLU(), nn.Linear(in_dimension, in_dimension))

    def forward(self, matrix, feature):
        re = self.ConvolutionLayer(matrix, feature)
        return self.LinearLayer(re), re


def train_gcn_ae(A, embedding_dim, lr, weight_decay, epochs, seed):
    """Train one GCN-AE on A (input and target) and return (R_e, final MSE)."""
    torch.manual_seed(seed)
    A = torch.as_tensor(A, dtype=torch.float32)
    model = GraphConvolution(A.shape[1], embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss(reduction="mean")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        rd, _ = model(A, A)
        loss = loss_fn(rd, A)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, re = model(A, A)
    return re.numpy(), float(loss)


# ----------------------------------------------------------------------------------------------
# VGAE with an MLP encoder
# ----------------------------------------------------------------------------------------------
class VGAE_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, A):
        h = self.hidden(A)                      # one aggregation over the row of A_k, no repeated propagation
        mu, log_var = self.mu(h), self.log_var(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var) if self.training else mu
        return z @ z.t(), mu, log_var           # reconstruction LOGITS (sigmoid inside the BCE)


def association_mask(n_lnc, n_dis, n_mi):
    """Boolean (N, N) mask of the lncRNA-disease, lncRNA-miRNA and miRNA-disease blocks (both directions)."""
    L, D, M = n_lnc, n_dis, n_mi
    mask = np.zeros((L + D + M, L + D + M), dtype=bool)
    mask[:L, L:] = True                 # lncRNA -> disease, miRNA
    mask[L:, :L] = True                 # disease, miRNA -> lncRNA
    mask[L:L + D, L + D:] = True        # disease -> miRNA
    mask[L + D:, L:L + D] = True        # miRNA -> disease
    return mask


def train_vgae_mlp(A, mask, hidden_dim, latent_dim, lr, epochs, kl_weight, seed):
    """Train one VGAE-MLP on A (fixed). Returns (mu, info) where info has the final losses and the
    reconstruction AUC on the masked (association) entries, as a training sanity check."""
    torch.manual_seed(seed)
    A = torch.as_tensor(A, dtype=torch.float32)
    mask_t = torch.as_tensor(mask)
    target = A[mask_t]
    n_pos = target.sum()
    pos_weight = (target.numel() - n_pos) / n_pos
    n = A.shape[0]
    model = VGAE_MLP(A.shape[1], hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, mu, log_var = model(A)
        recon = F.binary_cross_entropy_with_logits(logits[mask_t], target, pos_weight=pos_weight)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / n
        loss = recon + kl_weight * kl
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, mu, _ = model(A)
    auc = roc_auc_score(target.numpy(), logits[mask_t].numpy())
    return mu.numpy(), {"recon": float(recon), "kl": float(kl), "train_recon_auc": float(auc)}
