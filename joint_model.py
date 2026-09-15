import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from construct_multiview_gcn_gat import concatenate, MultiViewFeatureExtractor, reconstruct_similarity_matrix
from vgae_model import VGAE_Model
from LDAGM import LDAGM
import config


class JointVGAE_LDAGM(nn.Module):
    """
    Joint model combining GCN-Attention, VGAE for graph representation learning
    and LDAGM for link prediction in an end-to-end manner.
    """
    def __init__(self, num_lnc, num_diseases, num_mi, vgae_in_dim, vgae_hidden_dim, vgae_embed_dim,
                 ldagm_hidden_dim, ldagm_layers, drop_rate=config.DROP_RATE, use_aggregate=config.USE_AGGREGATE,
                 gcn_hidden_dim=config.GCN_HIDDEN_DIM, fusion_output_dim=config.FUSION_OUTPUT_DIM):
        super(JointVGAE_LDAGM, self).__init__()

        # Store dimensions for multi-view processing
        self.num_lnc = num_lnc
        self.num_diseases = num_diseases
        self.num_mi = num_mi

        # GCN-Attention components for multi-view feature extraction
        # Initialize feature extractors for each entity type in parallel
        def create_disease_extractor():
            return MultiViewFeatureExtractor(
                num_nodes=num_diseases, num_views=2,
                gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
            )

        def create_lnc_extractor():
            return MultiViewFeatureExtractor(
                num_nodes=num_lnc, num_views=2,
                gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
            )

        def create_mi_extractor():
            return MultiViewFeatureExtractor(
                num_nodes=num_mi, num_views=2,
                gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
            )

        # Parallel initialization of feature extractors
        with ThreadPoolExecutor(max_workers=3) as executor:
            self.disease_feature_extractor = executor.submit(create_disease_extractor).result()
            self.lnc_feature_extractor = executor.submit(create_lnc_extractor).result()
            self.mi_feature_extractor = executor.submit(create_mi_extractor).result()

        # VGAE component for graph representation learning
        self.vgae = VGAE_Model(vgae_in_dim, vgae_hidden_dim, vgae_embed_dim)

        # LDAGM component for link prediction
        # Input dimension: flattened features from pair_features tensor
        # pair_features shape: [batch_size, 2, features_per_node]
        # After flattening: [batch_size, 2 * features_per_node]
        # Actual feature dimension is 1024 based on runtime, so 512 per node # Based on A_encoder file shape
        network_num = config.NETWORK_NUM
        features_per_node = network_num * 2 * vgae_embed_dim
        ldagm_input_dim = 2 * features_per_node  # Fixed based on actual runtime dimensions
        self.ldagm = LDAGM(
            input_dimension=ldagm_input_dim,
            hidden_dimension=ldagm_hidden_dim,
            feature_num=1,  # Single feature vector per node pair
            hiddenLayer_num=ldagm_layers,
            drop_rate=drop_rate,
            use_aggregate=use_aggregate
        )

    def _extract_multiview_features(self, multi_view_data):
        """Run the three per-entity GCN-Attention feature extractors in parallel."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            disease_fused_features, _, _ = executor.submit(self.disease_feature_extractor, multi_view_data['disease']).result()
            lnc_fused_features, _, _ = executor.submit(self.lnc_feature_extractor, multi_view_data['lnc']).result()
            mi_fused_features, _, _ = executor.submit(self.mi_feature_extractor, multi_view_data['mi']).result()
        return disease_fused_features, lnc_fused_features, mi_fused_features

    def _reconstruct_similarities(self, disease_fused_features, lnc_fused_features, mi_fused_features):
        """Reconstruct per-entity similarity matrices from fused features in parallel."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            disease_similarity = executor.submit(reconstruct_similarity_matrix, disease_fused_features).result()
            lnc_similarity = executor.submit(reconstruct_similarity_matrix, lnc_fused_features).result()
            mi_similarity = executor.submit(reconstruct_similarity_matrix, mi_fused_features).result()
        return disease_similarity, lnc_similarity, mi_similarity

    def _build_adjacency_matrix(self, disease_similarity, lnc_similarity, mi_similarity,
                                 lnc_di_interaction, lnc_mi_interaction, mi_di_interaction):
        """Concatenate interaction matrices and reconstructed similarities into one adjacency matrix."""
        adj_matrix_np = concatenate(
            self.num_lnc, self.num_diseases, self.num_mi,
            lnc_di_interaction.detach().cpu().numpy(),
            lnc_mi_interaction.detach().cpu().numpy(),
            mi_di_interaction.detach().cpu().numpy(),
            lnc_similarity.detach().cpu().numpy(),
            disease_similarity.detach().cpu().numpy(),
            mi_similarity.detach().cpu().numpy()
        )
        return torch.tensor(adj_matrix_np, dtype=torch.float32, device=config.DEVICE)

    def _predict_links(self, mu, node_pairs, network_num, fold, dataset):
        """Build per-network pair features from VGAE embeddings and run LDAGM link prediction."""
        list_mu, avg_network_loss = encoder_matrix_by_vgae(dataset, self.vgae, network_num, fold)
        node1_computational_embeddings = mu[node_pairs[:, 0]]
        node2_computational_embeddings = mu[node_pairs[:, 1]]

        node1_all_features = []
        node2_all_features = []

        for i in range(network_num):
            node1_features_net_i = torch.cat([node1_computational_embeddings, list_mu[i][node_pairs[:, 0]]], dim=1)
            node2_features_net_i = torch.cat([node2_computational_embeddings, list_mu[i][node_pairs[:, 1]]], dim=1)

            node1_all_features.append(node1_features_net_i)
            node2_all_features.append(node2_features_net_i)

        node1_final_features = torch.cat(node1_all_features, dim=1)
        node2_final_features = torch.cat(node2_all_features, dim=1)

        pair_features = torch.stack([node1_final_features, node2_final_features], dim=1)
        pair_features_flattened = pair_features.view(pair_features.size(0), -1)
        link_predictions = self.ldagm(pair_features_flattened)
        return avg_network_loss, link_predictions

    def forward(self, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, node_pairs=None, network_num=4, fold=0, dataset="dataset1"):
        """
        Forward pass for joint end-to-end training.

        Args:
            multi_view_data: Dictionary containing multi-view adjacency matrices
                - 'disease_matrices': List of disease similarity matrices
                - 'lnc_matrices': List of lncRNA similarity matrices
                - 'mi_matrices': List of miRNA similarity matrices
            lnc_di_interaction: lncRNA-disease interaction matrix
            lnc_mi_interaction: lncRNA-miRNA interaction matrix
            mi_di_interaction: miRNA-disease interaction matrix
            node_pairs: Node pairs for link prediction (optional)
            network_num: Number of networks for A_encoder loading
            fold: Fold number for A_encoder file paths
            dataset: Dataset name for A_encoder file paths

        Returns:
            If node_pairs is None: (reconstructed_adj, mu, log_var)
            If node_pairs is provided: (reconstructed_adj, mu, log_var, avg_network_loss, link_predictions)
        """
        # Step 1: Extract multi-view features using GCN-Attention
        disease_fused_features, lnc_fused_features, mi_fused_features = self._extract_multiview_features(multi_view_data)

        # Step 2: Reconstruct similarity matrices from fused features in parallel
        disease_similarity, lnc_similarity, mi_similarity = self._reconstruct_similarities(
            disease_fused_features, lnc_fused_features, mi_fused_features
        )

        # Step 3: Concatenate into final adjacency matrix
        adj_matrix = self._build_adjacency_matrix(
            disease_similarity, lnc_similarity, mi_similarity,
            lnc_di_interaction, lnc_mi_interaction, mi_di_interaction
        )

        # Step 4: Add self-loops and create features
        num_nodes = adj_matrix.shape[0]
        features_input = torch.eye(num_nodes, device=config.DEVICE)

        # Step 5: VGAE forward pass
        reconstructed_adj, mu, log_var = self.vgae(adj_matrix, features_input)

        if node_pairs is None:
            return reconstructed_adj, mu, log_var

        # Step 6-7: Extract per-network features and run LDAGM link prediction
        avg_network_loss, link_predictions = self._predict_links(mu, node_pairs, network_num, fold, dataset)
        return reconstructed_adj, mu, log_var, avg_network_loss, link_predictions


def encoder_matrix_by_vgae(dataset, vgae_model, network_num, fold):
    """
    Extract encoder matrices (mu) from VGAE for each network.

    Args:
        dataset: Dataset name
        vgae_model: VGAE model instance
        network_num: Number of networks
        fold: Current fold number

    Returns:
        List of mu tensors from each network, and the average per-network encoder loss
    """
    list_mu = []
    total_encoder_loss = 0.0
    for i in range(network_num):
        A = torch.Tensor(np.load('./our_dataset/' + dataset + '/A/A_' + str(fold + 1) + '_' + str(i + 1) + '.npy'))
        adj_matrix = torch.tensor(A, dtype=torch.float32, device=config.DEVICE)
        num_nodes = adj_matrix.shape[0]
        features_input = torch.eye(num_nodes, device=config.DEVICE)
        reconstructed_adj, mu, log_var = vgae_model(adj_matrix, features_input)
        # Compute reconstruction loss for this network
        total_links = adj_matrix.sum().item()
        pos_weight = torch.tensor(float(num_nodes**2 - total_links) / max(total_links, 1), device=config.DEVICE)

        reconstruction_loss = F.binary_cross_entropy_with_logits(
            reconstructed_adj.view(-1),
            adj_matrix.view(-1),
            pos_weight=pos_weight
        )

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / num_nodes

        # Combine losses for this network
        network_loss = reconstruction_loss + config.KL_WEIGHT * kl_loss
        total_encoder_loss += network_loss
        avg_loss = total_encoder_loss / network_num
        list_mu.append(mu)
    return list_mu, avg_loss


def joint_loss_function(reconstructed_adj, original_adj, mu, log_var, avg_network_loss,
                       link_predictions, link_labels, num_nodes, pos_weight,
                       vgae_weight=1.0, link_weight=1.0, kl_weight=0.1):
    """
    Unified loss function for joint end-to-end training.

    Args:
        reconstructed_adj: Reconstructed adjacency matrix from VGAE
        original_adj: Original adjacency matrix
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        avg_network_loss: Average additional encoder loss from A_encoder matrices
        link_predictions: Link prediction scores from LDAGM
        link_labels: Ground truth link labels
        num_nodes: Number of nodes in the graph
        pos_weight: Weight for positive samples in reconstruction loss
        vgae_weight: Weight for VGAE reconstruction loss
        link_weight: Weight for link prediction loss
        kl_weight: Weight for KL divergence term

    Returns:
        Combined loss value and individual loss components
    """
    # VGAE reconstruction loss
    vgae_reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstructed_adj.view(-1),
        original_adj.view(-1),
        pos_weight=pos_weight
    )

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / num_nodes

    # Link prediction loss
    link_prediction_loss = F.binary_cross_entropy_with_logits(
        link_predictions, link_labels
    )

    # Combined loss
    total_loss = (
        vgae_weight * vgae_reconstruction_loss +
        kl_weight * kl_loss +
        avg_network_loss +
        link_weight * link_prediction_loss
    )

    loss_components = {
        'total': total_loss.item(),
        'vgae_reconstruction': vgae_reconstruction_loss.item(),
        'kl_divergence': kl_loss.item(),
        'avg_network_loss': avg_network_loss.item(),
        'link_prediction': link_prediction_loss.item(),
    }
    return total_loss, loss_components
