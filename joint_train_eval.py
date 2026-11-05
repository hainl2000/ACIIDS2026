import time

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import config
from joint_model import JointVGAE_LDAGM, joint_loss_function


def predict(model, dataset, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
            batch_size=config.EVAL_BATCH_SIZE, device=config.DEVICE):
    """Score every pair of `dataset` in eval mode. Returns (labels, probabilities).

    Eval mode is deterministic (no dropout, z = mu), so the batch size does not change the predictions;
    a large batch avoids recomputing the whole graph pipeline for every 16 pairs.
    """
    model.eval()
    model = model.to(device)
    lnc_di_interaction = lnc_di_interaction.to(device)
    lnc_mi_interaction = lnc_mi_interaction.to(device)
    mi_di_interaction = mi_di_interaction.to(device)

    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch_node_pairs, batch_labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch_node_pairs = batch_node_pairs.to(device)
            # (Phase-1 features of this fold are stored inside the model as a buffer)
            _, _, _, link_predictions, _ = model(multi_view_data, lnc_di_interaction, lnc_mi_interaction,
                                                 mi_di_interaction, batch_node_pairs)
            all_predictions.extend(torch.sigmoid(link_predictions).cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    return np.array(all_labels), np.array(all_predictions)


def joint_train(num_lnc, num_diseases, num_mi, train_dataset, multi_view_data,
               lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, phase1_features,
               val_dataset=None, fold=0, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, lr=config.LEARNING_RATE,
               weight_decay=config.WEIGHT_DECAY, device=config.DEVICE,
               vgae_weight=config.VGAE_WEIGHT, link_weight=config.LINK_WEIGHT, kl_weight=config.KL_WEIGHT,
               vgae_hidden_dim=config.VGAE_HIDDEN_DIM, vgae_embed_dim=config.VGAE_EMBED_DIM,
               ldagm_hidden_dim=config.LDAGM_HIDDEN_DIM, ldagm_layers=config.LDAGM_LAYERS,
               patience=config.EARLY_STOP_PATIENCE, eval_batch_size=config.EVAL_BATCH_SIZE):
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
        phase1_features: (K, N, F) offline GCN-AE embeddings for this fold (A_encoder/), fixed
        val_dataset: held-out training pairs for early stopping (None -> train for `epochs` epochs)
        fold: fold index (only used in log lines)
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
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
        patience: stop after this many epochs without a new best validation AUC
        eval_batch_size: batch size used for validation scoring

    Returns:
        Trained model (weights of the best validation epoch when val_dataset is given; the epoch number
        is stored in model.best_epoch) and the per-epoch history (losses, plus val_auc / val_aupr).
    """

    # Model parameters
    num_nodes = num_lnc + num_diseases + num_mi
    in_dimension = num_lnc + num_diseases + num_mi

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
        phase1_features=phase1_features,
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
    best_auc, best_epoch, best_state, epochs_without_improvement = -np.inf, epochs, None, 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch_node_pairs, batch_labels in dataloader:
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass with end-to-end GCN-Attention + VGAE + LDAGM
            reconstructed_adj, mu, log_var, link_predictions, adj_final = model(
                multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_node_pairs
            )

            # VGAE reconstruction target = A_final, the same matrix the VGAE encoded (Eq. 22).
            # Detached so the reconstruction loss cannot "move the target" toward the decoder output;
            # the feature extractors still get gradient through the VGAE input path.
            recon_target = adj_final.detach()
            # Standard VGAE positive weight w = (N^2 - sum A) / sum A, computed on the actual target
            # (A_final has dense similarity blocks, so it is not a sparse binary graph).
            target_mass = recon_target.sum()
            pos_weight = (num_nodes ** 2 - target_mass) / target_mass

            # Compute joint loss (Eq. 32)
            total_loss, loss_components = joint_loss_function(
                reconstructed_adj, recon_target, mu, log_var,
                link_predictions, batch_labels, num_nodes, pos_weight,
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
        avg_losses["epoch"] = epoch + 1

        # Validation + early stopping
        marker = ""
        if val_dataset is not None:
            val_labels, val_preds = predict(model, val_dataset, multi_view_data, lnc_di_interaction,
                                            lnc_mi_interaction, mi_di_interaction, eval_batch_size, device)
            avg_losses["val_auc"] = roc_auc_score(val_labels, val_preds)
            avg_losses["val_aupr"] = average_precision_score(val_labels, val_preds)
            if avg_losses["val_auc"] > best_auc:
                best_auc, best_epoch, epochs_without_improvement = avg_losses["val_auc"], epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                marker = " *best"
            else:
                epochs_without_improvement += 1
        loss_history.append(avg_losses)

        val_text = (f" | val AUC {avg_losses['val_auc']:.4f} AUPR {avg_losses['val_aupr']:.4f}{marker}"
                    if val_dataset is not None else "")
        line = (f"Fold {fold + 1} epoch {epoch + 1:3d}/{epochs}: total {avg_losses['total']:.4f}, "
                f"link {avg_losses['link_prediction']:.4f}, recon {avg_losses['vgae_reconstruction']:.4f}, "
                f"KL {avg_losses['kl_divergence']:.4f}{val_text} | {time.time() - epoch_start:.0f}s")
        print("  " + line, flush=True)
        with open(config.LOG_FILE, 'a') as f:
            f.write(line + "\n")

        if val_dataset is not None and epochs_without_improvement >= patience:
            stop = (f"Fold {fold + 1}: early stopping at epoch {epoch + 1} "
                    f"(no val AUC improvement for {patience} epochs); best epoch {best_epoch}, val AUC {best_auc:.4f}")
            print("  " + stop, flush=True)
            with open(config.LOG_FILE, 'a') as f:
                f.write(stop + "\n")
            break

    if best_state is not None:
        model.load_state_dict(best_state)   # restore the best validation epoch
    model.best_epoch = best_epoch
    return model, loss_history


def joint_test(model, test_dataset, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
               batch_size=config.EVAL_BATCH_SIZE, fold=0, device=config.DEVICE):
    """
    Test function for joint model with GCN-Attention integration.

    Args:
        model: Trained JointVGAE_LDAGM model
        test_dataset: Test dataset
        multi_view_data: Dictionary containing multi-view adjacency matrices
        lnc_di_interaction: lncRNA-disease interaction matrix
        lnc_mi_interaction: lncRNA-miRNA interaction matrix
        mi_di_interaction: miRNA-disease interaction matrix
        batch_size: Batch size for testing (does not change predictions, see predict())
        device: Testing device

    Returns:
        True labels and predictions
    """
    return predict(model, test_dataset, multi_view_data, lnc_di_interaction, lnc_mi_interaction,
                   mi_di_interaction, batch_size, device)
