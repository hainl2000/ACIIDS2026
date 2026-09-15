import torch
import numpy as np
from torch.utils.data import DataLoader
import config
from joint_model import JointVGAE_LDAGM, joint_loss_function


def joint_train(num_lnc, num_diseases, num_mi, train_dataset, multi_view_data,
               lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
               fold=0, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, lr=config.LEARNING_RATE,
               weight_decay=config.WEIGHT_DECAY, device=config.DEVICE,
               vgae_weight=config.VGAE_WEIGHT, link_weight=config.LINK_WEIGHT, kl_weight=config.KL_WEIGHT,
               vgae_hidden_dim=config.VGAE_HIDDEN_DIM, vgae_embed_dim=config.VGAE_EMBED_DIM,
               ldagm_hidden_dim=config.LDAGM_HIDDEN_DIM, ldagm_layers=config.LDAGM_LAYERS):
    """
    Joint end-to-end training function with GCN-Attention integration.
    Creates model internally and handles all training setup.

    Args:
        num_lnc: Number of lncRNA nodes
        num_diseases: Number of disease nodes
        num_mi: Number of miRNA nodes
        train_dataset: Training dataset
        multi_view_data: Dictionary containing multi-view adjacency matrices
        lnc_di_interaction: lncRNA-disease interaction matrix
        lnc_mi_interaction: lncRNA-miRNA interaction matrix
        mi_di_interaction: miRNA-disease interaction matrix
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Training device
        vgae_weight: Weight for VGAE loss
        link_weight: Weight for link prediction loss
        kl_weight: Weight for KL divergence
        vgae_hidden_dim: VGAE hidden dimension
        vgae_embed_dim: VGAE embedding dimension
        ldagm_hidden_dim: LDAGM hidden dimension
        ldagm_layers: Number of LDAGM layers

    Returns:
        Trained model and loss history
    """

    # Convert interactions to numpy for concatenation
    lnc_di_np = lnc_di_interaction.detach().cpu().numpy()
    lnc_mi_np = lnc_mi_interaction.detach().cpu().numpy()
    mi_di_np = mi_di_interaction.detach().cpu().numpy()

    # Model parameters
    num_nodes = num_lnc + num_diseases + num_mi
    in_dimension = num_lnc + num_diseases + num_mi
    total_links = (lnc_di_np.sum() + lnc_mi_np.sum() + mi_di_np.sum())*2 + num_lnc + num_diseases + num_mi
    pos_weight = torch.tensor(float(num_nodes**2 - total_links) / total_links, device=device)

    # Initialize joint model
    model = JointVGAE_LDAGM(
        num_lnc=num_lnc,
        num_diseases=num_diseases,
        num_mi=num_mi,
        vgae_in_dim=in_dimension,
        vgae_hidden_dim=vgae_hidden_dim,
        vgae_embed_dim=vgae_embed_dim,
        ldagm_hidden_dim=ldagm_hidden_dim,
        ldagm_layers=ldagm_layers,
        drop_rate=config.DROP_RATE,
        use_aggregate=config.USE_AGGREGATE,
        gcn_hidden_dim=config.GCN_HIDDEN_DIM,
        fusion_output_dim=config.FUSION_OUTPUT_DIM
    )

    model = model.to(device)
    lnc_di_interaction = lnc_di_interaction.to(device)
    lnc_mi_interaction = lnc_mi_interaction.to(device)
    mi_di_interaction = mi_di_interaction.to(device)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch_node_pairs, batch_labels in dataloader:
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass with end-to-end GCN-Attention + VGAE + LDAGM
            reconstructed_adj, mu, log_var, avg_network_loss, link_predictions = model(
                multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_node_pairs,
                network_num=config.NETWORK_NUM, fold=fold, dataset=config.DATASET
            )

            # Get the original adjacency matrix from the model's forward pass
            # We need to call the model again without node_pairs to get the original adj matrix
            with torch.no_grad():
                original_adj_reconstructed, _, _ = model(
                    multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, None,
                    network_num=config.NETWORK_NUM, fold=fold, dataset=config.DATASET
                )

            # Compute joint loss using the dynamically generated adjacency matrix
            total_loss, loss_components = joint_loss_function(
                reconstructed_adj, original_adj_reconstructed.detach(), mu, log_var,
                avg_network_loss, link_predictions, batch_labels, num_nodes, pos_weight,
                vgae_weight, link_weight, kl_weight
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(loss_components)

        # Average losses for the epoch
        avg_losses = {
            key: np.mean([loss[key] for loss in epoch_losses])
            for key in epoch_losses[0].keys()
        }
        loss_history.append(avg_losses)

        with open(config.LOG_FILE, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}: Total Loss: {avg_losses['total']:.4f}, Link Prediction: {avg_losses['link_prediction']:.4f}, VGAE Reconstruction: {avg_losses['vgae_reconstruction']:.4f}, KL Divergence: {avg_losses['kl_divergence']:.4f}, Average Network Loss: {avg_losses['avg_network_loss']:.4f}\n")
            f.write("-----------------------")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  Link Prediction: {avg_losses['link_prediction']:.4f}")
            print(f"  VGAE Reconstruction: {avg_losses['vgae_reconstruction']:.4f}")
            print(f"  KL Divergence: {avg_losses['kl_divergence']:.4f}")
            print(f"  Average Network Loss: {avg_losses['avg_network_loss']:.4f}")

    return model, loss_history


def joint_test(model, test_dataset, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_size=32, fold=0, device=config.DEVICE):
    """
    Test function for joint model with GCN-Attention integration.

    Args:
        model: Trained JointVGAE_LDAGM model
        test_dataset: Test dataset
        multi_view_data: Dictionary containing multi-view adjacency matrices
        lnc_di_interaction: lncRNA-disease interaction matrix
        lnc_mi_interaction: lncRNA-miRNA interaction matrix
        mi_di_interaction: miRNA-disease interaction matrix
        batch_size: Batch size for testing
        device: Testing device

    Returns:
        True labels and predictions
    """
    model.eval()
    model = model.to(device)
    lnc_di_interaction = lnc_di_interaction.to(device)
    lnc_mi_interaction = lnc_mi_interaction.to(device)
    mi_di_interaction = mi_di_interaction.to(device)

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_node_pairs, batch_labels in dataloader:
            batch_node_pairs = batch_node_pairs.to(device)

            # Forward pass with end-to-end GCN-Attention integration
            _, _, _, _, link_predictions = model(multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_node_pairs,
                                            network_num=config.NETWORK_NUM, fold=fold, dataset=config.DATASET)

            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(link_predictions)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    return np.array(all_labels), np.array(all_predictions)
