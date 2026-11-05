"""
Hyperparameter configuration for joint_end_to_end_main.py
"""

# Dataset and Fold
DATASET = "dataset2"
TOTAL_FOLDS = 5
NETWORK_NUM = 4      # N = 4 Phase-1 networks A_1..A_4 (A_encoder/A_{fold}_{k}.npy per fold)
A_ENCODER_DIM = 128  # Phase-1 GCN-AE embedding size (paper: 128-d), checked when loading A_encoder

# Device
DEVICE = "mps"  # "cuda" or "cpu"

# Model Hyperparameters (paper Sec. 3.3 / Fig. 9: best joint sweep setting)
GCN_HIDDEN_DIM = 512     # GCN hidden dimension 512
FUSION_OUTPUT_DIM = 128  # fusion output 128
VGAE_HIDDEN_DIM = 32     # VGAE hidden 32
VGAE_EMBED_DIM = 32      # VGAE embed 32
LDAGM_HIDDEN_DIM = 50    # MLP hidden dimension d_h = 50 (Fig. 8b)
LDAGM_LAYERS = 2         # MLP layers L = 2 (Fig. 8a)
DROP_RATE = 0.2          # Phase 2 dropout 0.2 (Sec. 3.1)
USE_AGGREGATE = True     # gated aggregation layers, Eq. 25-30

# Training Hyperparameters (paper Sec. 3.1 Phase 2, Fig. 7, Fig. 8c)
BATCH_SIZE = 16          # B = 16
EPOCHS = 5               # Fig. 7: best AUC/AUPR (0.9584/0.9590) at 5 epochs
LEARNING_RATE = 3e-4     # 3 x 10^-4
WEIGHT_DECAY = 2e-4      # 2 x 10^-4

# Loss Function Weights (paper: (lambda_recon, lambda_link, lambda_kl) = (0.9, 1.3, 0.3))
VGAE_WEIGHT = 0.9
LINK_WEIGHT = 1.3
KL_WEIGHT = 0.3

# Training Configuration
GRAD_CLIP = 1.0
EVAL_EVERY = 1
LR_PATIENCE = 5

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"