"""Các mô hình nơ-ron (tự cài bằng PyTorch thuần, không dùng thư viện đồ thị).

Đầu ra chung: logits (B, T, N, 3) cho từng rover ở từng bước. Mọi mô hình đều NHÂN QUẢ:
bước t chỉ dùng thông tin từ bước <= t (GRU một chiều, GAT trên từng bước).

  GRUNode  (ablation A, chỉ thời gian): GRU trên chuỗi thống kê bằng chứng của từng rover.
  GATSnap  (ablation C, chỉ đồ thị):    2 lớp GAT trên đồ thị liên lạc của TỪNG bước, không có bộ nhớ.
  GRUGAT   (mô hình chính, A + C):      GRU trên chuỗi bằng chứng của từng CẠNH (nhớ trôi tích luỹ),
                                         rồi 2 lớp GAT gộp bằng chứng từ mọi người quan sát
                                         (bắt được lỗi chỉ một số hàng xóm thấy, như spoof khớp range một phía).
  SeqAE    (baseline 3, không giám sát): autoencoder trên cửa sổ thống kê, chỉ học dữ liệu trung thực.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, drop=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, o))


class DenseGATLayer(nn.Module):
    """Lớp GAT có đặc trưng cạnh, kiểu GATv2, viết dạng ma trận đặc.
    h: (B', N, d) đặc trưng nút; ef: (B', N, N, de) đặc trưng cạnh [i, k]; mask: (B', N, N) bool [i, k].
    Nút k nhận thông điệp từ các i có mask[i, k] = True, trọng số attention học được."""

    def __init__(self, d, de, heads=2, drop=0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.src = nn.Linear(d, d)
        self.dst = nn.Linear(d, d)
        self.edge = nn.Linear(de, d)
        self.att = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.val = nn.Linear(d + de, d)
        self.out = mlp(2 * d, 2 * d, d, drop)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, ef, mask):
        Bp, N, d = h.shape
        z = self.src(h)[:, :, None, :] + self.dst(h)[:, None, :, :] + self.edge(ef)   # (B',N,N,d)
        z = F.leaky_relu(z, 0.2).view(Bp, N, N, self.h, self.dk)
        score = (z * self.att).sum(-1)                                                 # (B',N,N,H)
        score = score.masked_fill(~mask[..., None], float("-inf"))
        alpha = torch.softmax(score, dim=1)                                            # chuẩn hoá theo i
        alpha = torch.nan_to_num(alpha, nan=0.0)                                       # nút không có hàng xóm
        v = self.val(torch.cat([h[:, :, None, :].expand(Bp, N, N, d), ef], -1))
        v = v.view(Bp, N, N, self.h, self.dk)
        msg = (alpha[..., None] * v).sum(1).reshape(Bp, N, d)                          # (B',N,d)
        return self.norm(h + self.out(torch.cat([h, msg], -1)))


def masked_mean(x, mask, dim):
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)


class GRUNode(nn.Module):
    def __init__(self, hidden=64, n_node=6, **_):
        super().__init__()
        self.inp = nn.Linear(n_node, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = mlp(hidden, hidden, 3)

    def forward(self, batch):
        x = batch["node"]                                   # (B,T,N,6)
        B, T, N, Fn = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, Fn)
        h, _ = self.gru(torch.relu(self.inp(x)))
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        return self.head(h)


class GATSnap(nn.Module):
    def __init__(self, hidden=64, n_node=6, n_edge=3, layers=2, **_):
        super().__init__()
        self.edge_in = mlp(n_edge, hidden, hidden)
        self.node_in = nn.Linear(n_node, hidden)
        self.gat = nn.ModuleList([DenseGATLayer(hidden, hidden) for _ in range(layers)])
        self.head = mlp(hidden, hidden, 3)

    def forward(self, batch):
        ef, node, mask = batch["edge"], batch["node"], batch["valid"]
        B, T, N, _, _ = ef.shape
        ef = self.edge_in(ef).reshape(B * T, N, N, -1)
        mask = mask.reshape(B * T, N, N)
        h = self.node_in(node).reshape(B * T, N, -1) + masked_mean(ef, mask, 1)
        for layer in self.gat:
            h = layer(h, ef, mask)
        return self.head(h).reshape(B, T, N, 3)


class GRUGAT(nn.Module):
    def __init__(self, hidden=64, n_node=6, n_edge=3, layers=2, **_):
        super().__init__()
        self.edge_gru = nn.GRU(n_edge, hidden, batch_first=True)
        self.node_in = nn.Linear(n_node, hidden)
        self.node_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.gat = nn.ModuleList([DenseGATLayer(hidden, hidden) for _ in range(layers)])
        self.head = mlp(hidden, hidden, 3)

    def forward(self, batch):
        ef, node, mask = batch["edge"], batch["node"], batch["recent"]
        B, T, N, _, Fe = ef.shape
        x = ef.permute(0, 2, 3, 1, 4).reshape(B * N * N, T, Fe)
        he, _ = self.edge_gru(x)                                        # nhớ theo thời gian trên từng cạnh
        he = he.reshape(B, N, N, T, -1).permute(0, 3, 1, 2, 4)          # (B,T,N,N,d)
        xn = torch.relu(self.node_in(node)).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        hn, _ = self.node_gru(xn)
        hn = hn.reshape(B, N, T, -1).permute(0, 2, 1, 3)                # (B,T,N,d)
        he = he.reshape(B * T, N, N, -1)
        mask = mask.reshape(B * T, N, N)
        h = hn.reshape(B * T, N, -1) + masked_mean(he, mask, 1)
        for layer in self.gat:
            h = layer(h, he, mask)
        return self.head(h).reshape(B, T, N, 3)


class SeqAE(nn.Module):
    """Autoencoder trên cửa sổ (W bước x 6 thống kê) của một rover."""

    def __init__(self, hidden=64, window=30, n_node=6, **_):
        super().__init__()
        d = window * n_node
        self.enc = nn.Sequential(nn.Linear(d, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden // 2))
        self.dec = nn.Sequential(nn.Linear(hidden // 2, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, d))

    def forward(self, x):                                   # x: (M, W*6) đã chuẩn hoá
        return self.dec(self.enc(x))


MODELS = {"gru": GRUNode, "gat": GATSnap, "grugat": GRUGAT}
