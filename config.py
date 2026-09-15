"""
Hyperparameter configuration for joint_end_to_end_main.py
"""

# Dataset and Fold
DATASET = "dataset2"
TOTAL_FOLDS = 5
NETWORK_NUM = 4  # Number of networks for A_encoder loading
A_ENCODER_DIM = 128

# Device
DEVICE = "mps"  # "cuda" or "cpu"

# Model Hyperparameters
GCN_HIDDEN_DIM = 128
FUSION_OUTPUT_DIM = 128
VGAE_HIDDEN_DIM = 64
VGAE_EMBED_DIM = 128
LDAGM_HIDDEN_DIM = 128
LDAGM_LAYERS = 4
DROP_RATE = 0.5
USE_AGGREGATE = True

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 4.381527985330247e-05

# Loss Function Weights
VGAE_WEIGHT = 1.0
LINK_WEIGHT = 2.0
KL_WEIGHT = 0.05

# Training Configuration
GRAD_CLIP = 1.0
EVAL_EVERY = 1
LR_PATIENCE = 5

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"