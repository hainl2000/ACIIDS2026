import torch
import torch.nn as nn
import torch.nn.functional as F

# Laplace normalization
class LplsNorm(nn.Module):
    def __init__(self):
        super(LplsNorm, self).__init__()

    def forward(self, adjacency_matrix):
        degree = torch.sum(adjacency_matrix, dim=-1)
        diag = torch.diag(torch.pow(degree, -0.5))
        out = diag.mm(adjacency_matrix).mm(diag)
        return out

# convolutional
class Convolution(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Convolution, self).__init__()
        self.LinearLayer = nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.ReLU()
        )

    def forward(self, adjacency_matrix, feature):
        mf = feature + torch.mm(adjacency_matrix, feature)
        out = self.LinearLayer(mf)
        return out

class VGAE_Encoder(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, embedding_dimension):
        super(VGAE_Encoder, self).__init__()
        self.lpls_norm = LplsNorm()
        
        # Lớp GCN cơ sở
        self.base_gcn = Convolution(in_dimension, hidden_dimension)
        
        # Hai lớp tuyến tính để tạo ra mean và log_var
        self.gcn_mean = Convolution(hidden_dimension, embedding_dimension)
        self.gcn_log_var = Convolution(hidden_dimension, embedding_dimension)

    def forward(self, adj_matrix, features):
        # 1. Chuẩn hóa ma trận kề
        norm_adj = self.lpls_norm(adj_matrix)
        
        # 2. Đưa qua lớp GCN cơ sở
        hidden = self.base_gcn(norm_adj, features)
        
        # 3. Tính mean và log_var
        mean = self.gcn_mean(norm_adj, hidden)
        log_var = self.gcn_log_var(norm_adj, hidden)
        
        return mean, log_var


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        # Tái tạo ma trận kề bằng cách lấy z nhân với chuyển vị của nó
        z_t = z.t()
        adj_reconstructed = torch.mm(z, z_t)
        return torch.sigmoid(adj_reconstructed)

class VGAE_Model(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, embedding_dimension):
        super(VGAE_Model, self).__init__()
        self.encoder = VGAE_Encoder(in_dimension, hidden_dimension, embedding_dimension)
        self.decoder = InnerProductDecoder()

    def reparameterize(self, mean, log_var):
        # Thủ thuật tái tham số hóa
        if self.training:
            std = torch.exp(0.5 * log_var)
            # Lấy mẫu từ phân phối chuẩn N(0, 1)
            eps = torch.randn_like(std) 
            return eps.mul(std).add_(mean)
        else:
            # Khi kiểm tra hoặc suy luận, chỉ cần dùng mean
            return mean

    def forward(self, adj_matrix, features):
        # Mã hóa để lấy mean và log_var
        mean, log_var = self.encoder(adj_matrix, features)
        
        # Lấy mẫu z từ không gian tiềm ẩn
        z = self.reparameterize(mean, log_var)
        
        # Giải mã để tái tạo ma trận kề
        adj_reconstructed = self.decoder(z)
        
        return adj_reconstructed, mean, log_var

def vgae_loss_function(preds, labels, mean, log_var, num_nodes, pos_weight):
    # 1. Reconstruction Loss
    # Dùng Binary Cross Entropy với Logits để ổn định hơn
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        preds.view(-1), labels.view(-1), pos_weight=pos_weight
    )
    
    # 2. KL Divergence Loss
    # Công thức tính KL divergence giữa N(mean, var) và N(0, 1)
    kl_divergence = -0.5 / num_nodes * torch.mean(torch.sum(
        1 + 2 * log_var - mean.pow(2) - (2 * log_var).exp(), dim=1
    ))
    
    return reconstruction_loss + kl_divergence