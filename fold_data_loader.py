from dataclasses import dataclass
from typing import Optional

import copy
import numpy as np
import pandas as pd
import torch

import config
from joint_dataset import JointDataset


@dataclass
class FoldData:
    """Everything needed to train/test one cross-validation fold."""
    num_lnc: int
    num_diseases: int
    num_mi: int
    train_dataset: JointDataset
    val_dataset: Optional[JointDataset]  # held-out training pairs for early stopping (None if VAL_RATIO == 0)
    test_dataset: JointDataset
    multi_view_data: dict
    lnc_di_tensor: torch.Tensor
    lnc_mi_tensor: torch.Tensor
    mi_di_tensor: torch.Tensor
    phase1_features: torch.Tensor  # (NETWORK_NUM, N, A_ENCODER_DIM) offline GCN-AE embeddings R_e of A_1..A_K


def _load_fold_split_indices(dataset, fold):
    positive5foldsidx = np.load(
        "./our_dataset/" + dataset + "/index/positive5foldsidx.npy",
        allow_pickle=True,
    )
    negative5foldsidx = np.load(
        "./our_dataset/" + dataset + "/index/negative5foldsidx.npy",
        allow_pickle=True,
    )
    positive_ij = np.load("./our_dataset/" + dataset + "/index/positive_ij.npy")
    negative_ij = np.load("./our_dataset/" + dataset + "/index/negative_ij.npy")

    train_positive_ij = positive_ij[positive5foldsidx[fold]["train"]]
    train_negative_ij = negative_ij[negative5foldsidx[fold]["train"]]
    test_positive_ij = positive_ij[positive5foldsidx[fold]["test"]]
    test_negative_ij = negative_ij[negative5foldsidx[fold]["test"]]

    return positive_ij, positive5foldsidx, train_positive_ij, train_negative_ij, test_positive_ij, test_negative_ij


def _load_similarity_matrices(dataset, fold):
    di_semantic_similarity = np.load("./our_dataset/" + dataset + "/multi_similarities/di_semantic_similarity.npy")
    di_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_gip_similarity_fold_{fold+1}.npy")
    lnc_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_gip_similarity_fold_{fold+1}.npy")
    lnc_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_func_similarity_fold_{fold+1}.npy")
    mi_gip_similarity = np.load("./our_dataset/" + dataset + "/multi_similarities/mi_gip_similarity.npy")
    mi_func_similarity = np.load("./our_dataset/" + dataset + "/multi_similarities/mi_func_similarity.npy")
    return {
        'di_semantic': di_semantic_similarity,
        'di_gip': di_gip_similarity,
        'lnc_gip': lnc_gip_similarity,
        'lnc_func': lnc_func_similarity,
        'mi_gip': mi_gip_similarity,
        'mi_func': mi_func_similarity,
    }


def _load_interaction_matrices(dataset):
    lnc_di = pd.read_csv('./our_dataset/' + dataset + '/interaction/lnc_di.csv')
    lnc_di.set_index('0', inplace=True)
    lnc_di = lnc_di.values

    lnc_mi = pd.read_csv('./our_dataset/' + dataset + '/interaction/lnc_mi.csv', index_col='0').values

    mi_di = pd.read_csv('./our_dataset/' + dataset + '/interaction/mi_di.csv')
    mi_di.set_index('0', inplace=True)
    mi_di = mi_di.values

    return lnc_di, lnc_mi, mi_di


def _load_phase1_features(dataset, fold, network_num):
    """Load the Phase-1 nonlinear features (paper Eq. 11-14) computed OFFLINE by the GCN-AE.

    obtain_nonlinear_features.ipynb trains one GCN-AE per network A_k of this fold (test positives
    already masked in A/A_{fold+1}_{k}.npy) and saves its embedding R_e to A_encoder/A_{fold+1}_{k}.npy.
    These are fixed inputs: they get no gradient and add no loss term to the joint training.
    """
    features = [
        np.load(f"./our_dataset/{dataset}/A_encoder/A_{fold + 1}_{k + 1}.npy")
        for k in range(network_num)
    ]
    return torch.tensor(np.stack(features), dtype=torch.float32)  # (network_num, N, feature_dim)


def _split_validation(train_positive_ij, train_negative_ij, val_ratio, seed):
    """Hold out val_ratio of the training positives and of the training negatives (keeps the 1:1 ratio).

    Note: the held-out positives stay visible in the input graph (only test positives are masked in
    preprocessing), so this split is used only to choose the stopping epoch, never to report results.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for ij in (train_positive_ij, train_negative_ij):
        perm = rng.permutation(len(ij))
        n_val = int(round(val_ratio * len(ij)))
        parts.append((ij[perm[n_val:]], ij[perm[:n_val]]))
    (pos_fit, pos_val), (neg_fit, neg_val) = parts
    return pos_fit, neg_fit, pos_val, neg_val


def load_fold_data(dataset, fold, device, network_num=config.NETWORK_NUM,
                   val_ratio=config.VAL_RATIO, seed=config.SEED):
    """Load and assemble all data needed to train/test a single cross-validation fold."""
    (positive_ij, positive5foldsidx, train_positive_ij, train_negative_ij,
     test_positive_ij, test_negative_ij) = _load_fold_split_indices(dataset, fold)

    sim = _load_similarity_matrices(dataset, fold)
    lnc_di, lnc_mi, mi_di = _load_interaction_matrices(dataset)
    lnc_di_copy = copy.copy(lnc_di)

    # Get dimensions
    num_diseases = sim['di_semantic'].shape[0]
    num_lnc = sim['lnc_gip'].shape[0]
    num_mi = sim['mi_gip'].shape[0]
    lncRNALen = num_lnc

    # Remove test edges from training adjacency matrix
    for ij in positive_ij[positive5foldsidx[fold]['test']]:
        lnc_di_copy[ij[0], ij[1] - lncRNALen] = 0

    # Create adjacency matrices for each entity type
    disease_adjacency_matrices = [
        torch.tensor(sim['di_semantic'], dtype=torch.float32).to(device),
        torch.tensor(sim['di_gip'], dtype=torch.float32).to(device)
    ]
    lncRNA_adjacency_matrices = [
        torch.tensor(sim['lnc_gip'], dtype=torch.float32).to(device),
        torch.tensor(sim['lnc_func'], dtype=torch.float32).to(device)
    ]
    miRNA_adjacency_matrices = [
        torch.tensor(sim['mi_gip'], dtype=torch.float32).to(device),
        torch.tensor(sim['mi_func'], dtype=torch.float32).to(device)
    ]

    # Create datasets (optionally hold out a validation split from the training pairs for early stopping)
    val_dataset = None
    if val_ratio > 0:
        train_positive_ij, train_negative_ij, val_positive_ij, val_negative_ij = _split_validation(
            train_positive_ij, train_negative_ij, val_ratio, seed + fold)
        val_dataset = JointDataset(val_positive_ij, val_negative_ij, "val", dataset)
    train_dataset = JointDataset(train_positive_ij, train_negative_ij, "train", dataset)
    test_dataset = JointDataset(test_positive_ij, test_negative_ij, "test", dataset)

    # Prepare multi-view data structure for GCN-Attention
    multi_view_data = {
        'disease': disease_adjacency_matrices,
        'lnc': lncRNA_adjacency_matrices,
        'mi': miRNA_adjacency_matrices
    }

    # Prepare interaction matrices as tensors
    lnc_di_tensor = torch.tensor(lnc_di_copy, dtype=torch.float32).to(device)
    lnc_mi_tensor = torch.tensor(lnc_mi, dtype=torch.float32).to(device)
    mi_di_tensor = torch.tensor(mi_di, dtype=torch.float32).to(device)

    # Phase-1 offline GCN-AE embeddings for this fold
    phase1_features = _load_phase1_features(dataset, fold, network_num)
    expected = (network_num, num_lnc + num_diseases + num_mi, config.A_ENCODER_DIM)
    if tuple(phase1_features.shape) != expected:
        raise ValueError(f"A_encoder features have shape {tuple(phase1_features.shape)}, expected {expected}")
    phase1_features = phase1_features.to(device)

    return FoldData(
        num_lnc=num_lnc,
        num_diseases=num_diseases,
        num_mi=num_mi,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        multi_view_data=multi_view_data,
        lnc_di_tensor=lnc_di_tensor,
        lnc_mi_tensor=lnc_mi_tensor,
        mi_di_tensor=mi_di_tensor,
        phase1_features=phase1_features,
    )
