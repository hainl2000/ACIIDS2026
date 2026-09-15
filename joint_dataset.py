import torch
import numpy as np
from torch.utils.data import Dataset


class JointDataset(Dataset):
    """
    Dataset for joint end-to-end training that provides both graph structure
    and node pair features for link prediction.
    """
    def __init__(self, positive_ij, negative_ij, mode, dataset):
        super().__init__()

        # Combine positive and negative samples
        all_ij = np.vstack([positive_ij, negative_ij])
        labels = np.hstack([np.ones(len(positive_ij)), np.zeros(len(negative_ij))])

        self.node_pairs = torch.tensor(all_ij, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        print(f"{dataset} {mode} dataset loaded with {len(self.node_pairs)} samples")

    def __getitem__(self, index):
        return self.node_pairs[index], self.labels[index]

    def __len__(self):
        return len(self.node_pairs)
