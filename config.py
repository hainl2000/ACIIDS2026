"""Hyper-parameters of the final model: masked VGAE -> LDAGM, blended with a gradient-boosting stack."""

# Evaluation splits (our_dataset/<folder>): folder -> name. Fu = folder dataset1 (paper "Dataset 2"), Zhou = folder
# dataset2 (paper "Dataset 1"). The Fu random split is not used: its labels are propagated to Disease-Ontology ancestors,
# so random test positives are implied by training ones; the leaf split (P2x) removes that leak.
SPLITS = {
    "dataset2_v2026": "Zhou random",
    "dataset2_v2026_fam": "Zhou disease family",
    "dataset2_v2026_cs": "Zhou cold-start lncRNA",
    "dataset1_v2026_leaf": "Fu leaf (P2x)",
    "dataset1_v2026_fam": "Fu disease family",
    "dataset1_v2026_cs": "Fu cold-start lncRNA",
}
TOTAL_FOLDS = 5
SEED = 42                  # validation split and every per-fold seed
VAL_RATIO = 0.1            # share of the training positives / negatives held out (without a group-aware split)
N_SEEDS = 5                # model seeds per split (seed s changes the VGAE / LDAGM seeds and the cross-fitting chunks)

# Masked VGAE encoder: rows of A (similarity blocks + association blocks) -> 256 -> (mu, log_var) of size 128
VGAE_HIDDEN_DIM = 256
EMBED_DIM = 128
VGAE_LR = 1e-3
VGAE_EPOCHS = 200
VGAE_KL_WEIGHT = 0.3 / 0.9
MASK_EDGE_RATE = 0.5       # share of the association edges hidden from the input (and scored) every epoch
MASK_ROW_RATE = 0.1        # share of the lncRNAs whose lncRNA-disease row is hidden every epoch

# LDAGM on z-scored mu (pair = [mu_l || mu_d]), early stopping on validation AUC
LDAGM_HIDDEN_DIM = 50
LDAGM_LAYERS = 2
DROP_RATE = 0.2
USE_AGGREGATE = True
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-4
EPOCHS = 100
EARLY_STOP_PATIENCE = 10

# Stack: 13 pair features, cross-fitted on 5 chunks that follow the split's grouping
CROSS_FIT_CHUNKS = 5
KRONRLS_SIGMAS = (0.01, 0.1, 1.0, 10.0, 100.0)   # KronRLS regularisation, chosen on validation AUC
GBM = dict(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, early_stopping=True)

# Blend: w z(LDAGM) + (1 - w) z(stack), w on this grid by validation AUC
BLEND_GRID = 11
