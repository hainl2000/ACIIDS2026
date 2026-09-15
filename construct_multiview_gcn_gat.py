import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd

# Device configuration for CUDA support
def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device in construct multiview GCN GAT: {DEVICE}")
# =============================================================================
# Phần 1: GCN Feature Learning (Theo Eq. 7)
# =============================================================================
class GCNBlock(nn.Module):
    """
    Một khối GCN 2 lớp độc lập cho một view, bám sát mô tả 2.3.1.
    Sử dụng GCNConv của PyG để triển khai đúng công thức (7).
    """
    def __init__(self, in_channels, hidden_channels=128, dropout_rate=0.1):
        super(GCNBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj_matrix):
        # Ensure tensors are on the correct device
        x = x.to(DEVICE)
        adj_matrix = adj_matrix.to(DEVICE)
        
        # Chuyển ma trận kề dày sang định dạng edge_index của PyG
        edge_index, _ = dense_to_sparse(adj_matrix)
        edge_index = edge_index.to(DEVICE)
        
        # Lớp 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Lớp 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return x

# =============================================================================
# Phần 2: Attention Fusion (Theo Eq. 8 và 9)
# =============================================================================
# class AttentionFusionWithCNN(nn.Module):
#     """
#     Triển khai chính xác cơ chế attention và tổng hợp bằng CNN 2D.
#     - Eq. (8): ωd = σ(fcd(AvgPool(Hd)))
#     - Eq. (9): Hf_d = CNNd(σ(ωd⋅Hd))
#     """
#     def __init__(self, num_views, num_nodes, hidden_dim=128, attention_dim=64, 
#                  fusion_output_dim=128, cnn_out_channels=32, kernel_size=3):
#         super(AttentionFusionWithCNN, self).__init__()
        
#         # --- Triển khai Eq. 8: Tính trọng số Attention ---
#         self.attention_fc = nn.Sequential(
#             nn.Linear(hidden_dim, attention_dim),
#             nn.Tanh(),
#             nn.Linear(attention_dim, 1)
#         )
        
#         # --- Triển khai Eq. 9: Tổng hợp bằng CNN ---
#         # Lớp CNN 2D. Input sẽ được reshape thành (batch, channels, height, width)
#         # Ta coi: batch=1, channels=1, height=num_views, width=num_nodes * hidden_dim
#         self.conv2d = nn.Conv2d(
#             in_channels=1, 
#             out_channels=cnn_out_channels, 
#             kernel_size=(kernel_size, kernel_size), 
#             padding='same'
#         )

#         # Lớp MLP cuối để map feature về đúng kích thước output
#         # Kích thước sau Conv2d là (cnn_out_channels, num_views, num_nodes * hidden_dim)
#         # Ta cần đưa nó về (num_nodes, fusion_output_dim)
#         final_mlp_in_dim = cnn_out_channels * num_views * hidden_dim
#         self.final_mlp = nn.Sequential(
#             nn.Linear(final_mlp_in_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, fusion_output_dim)
#         )
        
#         self.num_nodes = num_nodes

#     def forward(self, view_embeddings_list):
#         # Input: list các tensor shape (num_nodes, hidden_dim)
#         # Stack lại thành (num_views, num_nodes, hidden_dim)
#         stacked_embeddings = torch.stack(view_embeddings_list, dim=0)
#         num_views, num_nodes, hidden_dim = stacked_embeddings.shape

#         # --- Bước 1: Tính Attention Weights (Eq. 8) ---
#         # AvgPool(Hd): Lấy trung bình theo chiều các node
#         # Shape: (num_views, hidden_dim)
#         view_summaries = torch.mean(stacked_embeddings, dim=1)
        
#         # fcd(AvgPool(Hd)): Đưa qua mạng FC để có score
#         # Shape: (num_views, 1) -> (num_views,)
#         attention_scores = self.attention_fc(view_summaries).squeeze(-1)
        
#         # σ(...): Dùng Softmax để có trọng số
#         # Shape: (num_views,)
#         attention_weights = F.softmax(attention_scores, dim=0)
        
#         # --- Bước 2: Áp dụng Attention và CNN (Eq. 9) ---
#         # ωd ⋅ Hd: Nhân trọng số attention
#         # Reshape weights: (num_views, 1, 1) để nhân broadcasting
#         weighted_embeddings = stacked_embeddings * attention_weights.view(num_views, 1, 1)
        
#         # σ(ωd ⋅ Hd): Áp dụng activation
#         activated_embeddings = F.relu(weighted_embeddings)

#         # Chuẩn bị cho CNNd: Reshape tensor
#         # Shape: (num_views, num_nodes, hidden_dim) -> (1, 1, num_views, num_nodes * hidden_dim)
#         # Đây là một cách diễn giải để sử dụng Conv2d
#         conv_input = activated_embeddings.view(1, 1, num_views, num_nodes * hidden_dim)
        
#         # CNNd(...): Đưa qua lớp Conv2d
#         # Shape: (1, cnn_out_channels, num_views, num_nodes * hidden_dim)
#         conv_output = self.conv2d(conv_input)
        
#         # Đưa về shape cuối cùng
#         # Flatten: (1, cnn_out_channels * num_views * num_nodes * hidden_dim)
#         flattened_output = torch.flatten(conv_output)
        
#         # Reshape và đi qua MLP cuối
#         # Chúng ta cần reshape lại để mỗi node có feature riêng
#         # (num_nodes, cnn_out_channels * num_views * hidden_dim)
#         reshaped_for_mlp = conv_output.permute(0, 2, 3, 1).reshape(num_nodes, -1)
        
#         # (num_nodes, fusion_output_dim)
#         fused_features = self.final_mlp(reshaped_for_mlp)
        
#         return fused_features, attention_weights

class AttentionFusion(nn.Module):
    """
    Phiên bản sửa lỗi, không dùng Conv2D.
    Nó tính toán trọng số attention cho mỗi view và sau đó tổng hợp
    các embedding bằng một tổng có trọng số, rồi đưa qua một lớp MLP.
    """
    def __init__(self, num_views, num_nodes, hidden_dim=128, attention_dim=64, 
                 fusion_output_dim=128):
        super(AttentionFusion, self).__init__()
        
        # --- Tính trọng số Attention cho mỗi view ---
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # --- Lớp MLP để tổng hợp các embedding đã được gán trọng số ---
        # Input là embedding được ghép nối từ các view
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_views, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, fusion_output_dim)
        )

    def forward(self, view_embeddings_list):
        stacked_embeddings = torch.stack(view_embeddings_list, dim=0)
        num_views, num_nodes, hidden_dim = stacked_embeddings.shape
        # --- Bước 1: Tính Attention Weights cho mỗi view ---
        view_summaries = torch.mean(stacked_embeddings, dim=1)
        attention_scores = self.attention_fc(view_summaries).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # --- Bước 2: Áp dụng Attention và tổng hợp bằng MLP ---
        # Nhân trọng số attention
        weighted_embeddings = stacked_embeddings * attention_weights.view(num_views, 1, 1)
        # Thay vì đưa qua Conv2D, ta ghép nối các embedding đã được gán trọng số
        # và đưa qua một MLP để học cách kết hợp chúng.
        # Permute để có shape (num_nodes, num_views, hidden_dim)
        # rồi reshape thành (num_nodes, num_views * hidden_dim)
        fusion_input = weighted_embeddings.permute(1, 0, 2).reshape(num_nodes, -1)
        
        fused_features = self.fusion_mlp(fusion_input)
        return fused_features, attention_weights

# =============================================================================
# Phần 3: Model Tổng Thể
# =============================================================================
class MultiViewFeatureExtractor(nn.Module):
    # def __init__(self, num_nodes, num_views, gcn_hidden_dim=128, fusion_output_dim=128):
    #     super(MultiViewFeatureExtractor, self).__init__()
        
    #     # Đặc trưng đầu vào là ma trận đơn vị
    #     self.initial_features = nn.Parameter(torch.eye(num_nodes), requires_grad=False)
        
    #     # Các khối GCN độc lập cho mỗi view
    #     self.gcn_blocks = nn.ModuleList([
    #         GCNBlock(in_channels=num_nodes, hidden_channels=gcn_hidden_dim)
    #         for _ in range(num_views)
    #     ])
        
    #     # Module tổng hợp feature bằng Attention và CNN
    #     self.fusion_module = AttentionFusionWithCNN(
    #         num_views=num_views,
    #         num_nodes=num_nodes,
    #         hidden_dim=gcn_hidden_dim,
    #         fusion_output_dim=fusion_output_dim
    #     )

    def __init__(self, num_nodes, num_views, gcn_hidden_dim=128, fusion_output_dim=128):
        super(MultiViewFeatureExtractor, self).__init__()
        
        self.initial_features = nn.Parameter(torch.eye(num_nodes), requires_grad=False)
        
        self.gcn_blocks = nn.ModuleList([
            GCNBlock(in_channels=num_nodes, hidden_channels=gcn_hidden_dim)
            for _ in range(num_views)
        ])
        
        # SỬA Ở ĐÂY: Đổi tên class thành AttentionFusion
        self.fusion_module = AttentionFusion(
            num_views=num_views,
            num_nodes=num_nodes,
            hidden_dim=gcn_hidden_dim,
            fusion_output_dim=fusion_output_dim
        )
        
        # Move model to device
        self.to(DEVICE)

    def forward(self, adjacency_matrices_list):
        # Ensure adjacency matrices are on the correct device
        adjacency_matrices_list = [adj.to(DEVICE) for adj in adjacency_matrices_list]
        
        # 1. Trích xuất feature từ mỗi view bằng GCN
        view_embeddings = [
            gcn(self.initial_features, adj) 
            for gcn, adj in zip(self.gcn_blocks, adjacency_matrices_list)
        ]
        
        # 2. Tổng hợp các feature bằng module fusion
        fused_features, attention_weights = self.fusion_module(view_embeddings)
        
        return fused_features, attention_weights, view_embeddings


def reconstruct_similarity_matrix(embeddings):
    """
    Tái tạo lại ma trận tương đồng từ các vector embedding.
    
    Args:
        embeddings (torch.Tensor): Ma trận embedding có shape (num_nodes, embedding_dim).
        
    Returns:
        torch.Tensor: Ma trận tương đồng mới có shape (num_nodes, num_nodes)
                      với các giá trị trong khoảng [0, 1].
    """
    # Đảm bảo đầu vào là tensor PyTorch
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
        
    # Tính tích vô hướng: H * H^T
    # torch.matmul thực hiện phép nhân ma trận
    dot_product = torch.matmul(embeddings, embeddings.T)
    
    # Áp dụng hàm Sigmoid để chuẩn hóa giá trị về khoảng [0, 1]
    similarity_matrix = torch.sigmoid(dot_product)
    
    return similarity_matrix    


def concatenate(lnclen, dilen, milen, lnc_di, lnc_mi, mi_di, lncSiNet, diSiNet, miSiNet):
    A = np.zeros((lnclen + dilen + milen, lnclen + dilen + milen))
    A[: lnclen, lnclen: lnclen + dilen] = lnc_di
    A[lnclen: lnclen + dilen, : lnclen] = lnc_di.T
    A[: lnclen, lnclen + dilen: ] = lnc_mi
    A[lnclen + dilen: , : lnclen] = lnc_mi.T
    A[lnclen: lnclen + dilen, lnclen + dilen: ] = mi_di.T
    A[lnclen + dilen: , lnclen: lnclen + dilen] = mi_di
    A[: lnclen, : lnclen] = lncSiNet
    A[lnclen: lnclen + dilen, lnclen: lnclen + dilen] = diSiNet
    A[lnclen + dilen: , lnclen + dilen: ] = miSiNet
    return A