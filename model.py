"""
model.py 
Architecture:
  (A) Soft Temporal Decomposition Branch: detects phase boundaries from motion gradients
  (B) Diagnostic Branch with Adaptive Adjacency: extracts spatial-temporal features
  (C) MIL Gated Attention Aggregation (PIW): aggregates temporal instances for classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import Graph

# Model / data constants
TARGET_LENGTH = 100
NUM_KEYPOINTS = 17
NUM_PHASES    = 3
PHASE_NAMES   = ['EarlyTurn', 'MidTurn', 'LateTurn']

ADJ_INIT_TEMPERATURE = 2.0  # annealed toward ADJ_FINAL_TEMPERATURE during training

TRUNK_JOINTS      = [0, 7, 8, 9, 11, 14]
LOWER_BODY_JOINTS = [1, 2, 3, 4, 5, 6, 8]

EDGE_GROUPS = {
    'Lower Limbs': [(1, 2), (2, 3), (4, 5), (5, 6)],
    'Trunk':       [(0, 7), (7, 8), (8, 1), (8, 4), (1, 4), (0, 9)],
    'Upper Limbs': [(11, 12), (12, 13), (14, 15), (15, 16)],
    'Head-Neck':   [(9, 10), (8, 11), (8, 14)],
}


# ST-GCN building blocks
class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1))

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A_size, temporal_kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, A_size)
        padding = (temporal_kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel_size, 1),
                      stride=(stride, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.bn      = nn.BatchNorm2d(out_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, A):
        res = self.residual(x)
        x   = self.gcn(x, A)
        x   = self.bn(x)
        x   = self.tcn(x)
        x   = x + res
        x   = self.relu(x)
        x   = self.dropout(x)
        return x


# Soft Temporal Decomposition Branch
class SoftTemporalDecomp(nn.Module):
    """
    Detects phase boundaries from motion gradients via a lightweight ST-GCN encoder
    and a temporal convolutional boundary predictor. Trained exclusively by auxiliary
    regularization losses (sparsity + smoothness), decoupled from the classification
    objective to preserve clinical interpretability of the discovered phase structure.
    """

    def __init__(self, in_channels, hidden_dim, A, num_boundaries=2, dropout=0.5):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.hidden_dim     = hidden_dim
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)

        self.encoder = nn.ModuleList([
            STGCNBlock(in_channels, hidden_dim, spatial_kernel_size, 9, 1, dropout),
            STGCNBlock(hidden_dim,  hidden_dim, spatial_kernel_size, 9, 1, dropout),
        ])
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(A.size())) for _ in range(2)
        ])

        self.boundary_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim,      kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim,     hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1,              kernel_size=1),
        )

        self.position_prior = nn.Parameter(self._create_position_prior())

    def _create_position_prior(self):
        t     = torch.linspace(0, 1, TARGET_LENGTH)
        prior = torch.exp(-((t - 0.3) ** 2) / 0.03) + torch.exp(-((t - 0.7) ** 2) / 0.03)
        return prior / prior.max() * 0.3

    def compute_differential(self, x):
        diff     = x[:, :, 1:] - x[:, :, :-1]
        zero_pad = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
        return torch.cat([zero_pad, diff], dim=2)

    def detect_peaks(self, boundary_scores):
        N, T   = boundary_scores.size()
        padded = F.pad(boundary_scores, (1, 1), mode='replicate')
        is_peak = (boundary_scores >= padded[:, :-2]) & (boundary_scores >= padded[:, 2:])
        peak_scores = boundary_scores * is_peak.float()
        _, top_indices = torch.topk(peak_scores, self.num_boundaries, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)
        return top_indices.float() / (T - 1)

    def _positions_to_phases(self, boundary_positions, T, device):
        N  = boundary_positions.size(0)
        t  = torch.linspace(0, 1, T, device=device).unsqueeze(0)
        b1 = boundary_positions[:, 0:1]
        b2 = boundary_positions[:, 1:2]
        phase_assignments = torch.zeros(N, self.num_boundaries + 1, T, device=device)
        phase_assignments[:, 0, :] = torch.sigmoid((b1 - t) * 5)
        phase_assignments[:, 1, :] = torch.sigmoid((t - b1) * 5) * torch.sigmoid((b2 - t) * 5)
        phase_assignments[:, 2, :] = torch.sigmoid((t - b2) * 5)
        phase_assignments = phase_assignments / (phase_assignments.sum(dim=1, keepdim=True) + 1e-8)
        return phase_assignments

    def compute_boundary_loss(self, boundary_scores):
        sparsity_loss   = boundary_scores.mean()
        smoothness_loss = ((boundary_scores[:, 1:] - boundary_scores[:, :-1]) ** 2).mean()
        return sparsity_loss, smoothness_loss

    def forward(self, x, return_scores=False):
        N, C, T, V = x.size()

        for i, layer in enumerate(self.encoder):
            x = layer(x, self.A * self.edge_importance[i])

        x_temporal  = x.mean(dim=3)
        x_diff      = self.compute_differential(x_temporal)
        x_combined  = torch.cat([x_temporal, x_diff], dim=1)

        boundary_logits   = self.boundary_predictor(x_combined).squeeze(1)
        boundary_logits   = boundary_logits + self.position_prior.unsqueeze(0)
        boundary_scores   = torch.sigmoid(boundary_logits)
        boundary_positions = self.detect_peaks(boundary_scores)
        phase_assignments  = self._positions_to_phases(boundary_positions, T, x.device)

        if return_scores:
            return phase_assignments, boundary_positions, boundary_scores
        return phase_assignments, boundary_positions



# Adaptive Adjacency (learnable graph topology)
class AdaptiveAdjacency(nn.Module):
    """
    Modulates the skeleton adjacency matrix via a soft importance mask
    M = sigmoid(E + P_motor), where E is a learnable matrix and P_motor
    is an anatomical prior emphasizing lower-limb and trunk joint pairs.
    """

    def __init__(self, num_joints=17, init_temperature=2.0):
        super().__init__()
        self.num_joints  = num_joints
        self.edge_logits = nn.Parameter(torch.zeros(num_joints, num_joints))
        self.temperature = init_temperature
        self.register_buffer('anatomical_prior', self._create_anatomical_prior())

    def _create_anatomical_prior(self):
        prior = torch.zeros(self.num_joints, self.num_joints)
        # Lower-limb–trunk cross-connections: 0.5 (set first)
        for i in LOWER_BODY_JOINTS:
            for j in TRUNK_JOINTS:
                if i < self.num_joints and j < self.num_joints:
                    prior[i, j] = 0.5
                    prior[j, i] = 0.5
        # Lower-limb joint pairs: 1.0 (overrides any overlap above)
        for i in LOWER_BODY_JOINTS:
            for j in LOWER_BODY_JOINTS:
                if i < self.num_joints and j < self.num_joints:
                    prior[i, j] = 1.0
        return prior

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_importance_mask(self):
        return torch.sigmoid((self.edge_logits + self.anatomical_prior) / self.temperature)

    def forward(self, A, return_mask=False):
        mask       = self.get_importance_mask()
        A_sum      = A.sum(dim=0) if A.dim() == 3 else A
        edge_exists    = (A_sum > 0).float()
        effective_mask = mask * edge_exists + (1 - edge_exists)
        if A.dim() == 3:
            K         = A.size(0)
            A_weighted = A * effective_mask.unsqueeze(0).expand(K, -1, -1)
        else:
            A_weighted = A * effective_mask
        if return_mask:
            return A_weighted, mask
        return A_weighted

    def compute_sparsity_loss(self):
        return self.get_importance_mask().mean()

    def get_important_edges(self, top_k=10):
        mask  = self.get_importance_mask().detach().cpu()
        edges = []
        for i in range(self.num_joints):
            for j in range(i + 1, self.num_joints):
                edges.append((i, j, max(mask[i, j].item(), mask[j, i].item())))
        edges.sort(key=lambda x: x[2], reverse=True)
        return edges[:top_k]

    def get_edge_group_importance(self):
        mask = self.get_importance_mask().detach().cpu()
        return {
            name: float(np.mean([max(mask[i, j].item(), mask[j, i].item())
                                 for i, j in edges
                                 if i < self.num_joints and j < self.num_joints]) or 0.0)
            for name, edges in EDGE_GROUPS.items()
        }

    def get_joint_importance_from_edges(self):
        mask = self.get_importance_mask().detach().cpu()
        joint_importance = torch.zeros(self.num_joints)
        for i in range(self.num_joints):
            connected = mask[i, :].tolist() + mask[:, i].tolist()
            joint_importance[i] = float(np.mean([e for e in connected if e > 0]))
        return joint_importance


# Temporal Turning Attention
class TemporalTurningAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = in_channels // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q_proj    = nn.Linear(in_channels, in_channels)
        self.k_proj    = nn.Linear(in_channels, in_channels)
        self.v_proj    = nn.Linear(in_channels, in_channels)
        self.dropout   = nn.Dropout(dropout)
        self.turning_query = nn.Parameter(torch.randn(1, 1, in_channels))

    def forward(self, x, return_attention=False):
        N, C, T, V = x.size()
        x_temporal = x.mean(dim=3).permute(0, 2, 1)
        turning_q  = self.turning_query.expand(N, -1, -1)
        Q      = self.q_proj(turning_q)
        K      = self.k_proj(x_temporal)
        V_feat = self.v_proj(x_temporal)
        Q      = Q.view(N, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K      = K.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        V_feat = V_feat.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn   = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attn   = self.dropout(attn)
        temporal_attention = attn.mean(dim=1).squeeze(1)
        attn_weights = temporal_attention.unsqueeze(1).unsqueeze(3)
        attended_x   = x * (1 + attn_weights)
        if return_attention:
            return attended_x, temporal_attention
        return attended_x


# PIW: Phase Importance Weighting (MIL gated attention aggregation)
class PhaseImportanceWeighting(nn.Module):
    """
    Gated attention aggregation over temporal instances:
      alpha_k = softmax( w^T ( tanh(W1 z_k) * sigmoid(W2 z_k) ) )
    """

    def __init__(self, feature_dim, hidden_dim=128, num_phases=3):
        super().__init__()
        self.attention_V       = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.attention_U       = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, phase_features, return_attention=False):
        V      = self.attention_V(phase_features)
        U      = self.attention_U(phase_features)
        gated  = V * U
        scores = self.attention_weights(gated).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        bag_feature = torch.sum(weights.unsqueeze(-1) * phase_features, dim=1)
        if return_attention:
            return bag_feature, weights
        return bag_feature



# main model

class TempoDecompGCN(nn.Module):
    """
    Weakly-Supervised Temporal Decomposition GCN for Parkinsonian Turning Assessment.
    Dual-branch architecture:
      - SoftTemporalDecomp branch: soft temporal decomposition (decoupled from classification loss)
      - Diagnostic branch: ST-GCN with adaptive adjacency
    Aggregated via PIW (gated attention MIL) for video-level classification.
    """

    def __init__(self, in_channels=6, hidden_dim=64, num_classes=2,
                 num_phases=3, dropout=0.5):
        super().__init__()
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim

        graph = Graph(layout='17new', strategy='spatial', max_hop=1, dilation=1)
        A     = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_KEYPOINTS)

        self.soft_decomp = SoftTemporalDecomp(
            in_channels, hidden_dim, A,
            num_boundaries=num_phases - 1,
            dropout=dropout
        )

        self.adaptive_adj = AdaptiveAdjacency(NUM_KEYPOINTS, ADJ_INIT_TEMPERATURE)

        self.class_backbone = nn.ModuleList([
            STGCNBlock(in_channels,      hidden_dim,     spatial_kernel_size, 9, 1, dropout),
            STGCNBlock(hidden_dim,       hidden_dim,     spatial_kernel_size, 9, 1, dropout),
            STGCNBlock(hidden_dim,       hidden_dim,     spatial_kernel_size, 9, 1, dropout),
            STGCNBlock(hidden_dim,       hidden_dim,     spatial_kernel_size, 9, 1, dropout),
            STGCNBlock(hidden_dim,       hidden_dim,     spatial_kernel_size, 9, 1, dropout),
            STGCNBlock(hidden_dim,       hidden_dim * 2, spatial_kernel_size, 9, 1, dropout),
        ])
        self.class_edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(A.size())) for _ in range(6)
        ])

        self.temporal_attention = TemporalTurningAttention(hidden_dim * 2, num_heads=4, dropout=0.1)

        self.phase_transforms = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            ) for _ in range(num_phases)
        ])

        self.piw = PhaseImportanceWeighting(hidden_dim * 2, hidden_dim, num_phases)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def set_adj_temperature(self, temperature):
        self.adaptive_adj.set_temperature(temperature)

    def _get_weighted_adjacency(self, edge_importance):
        return self.adaptive_adj(self.A * edge_importance)

    def forward(self, x, return_analysis=False, return_losses=False):
        N, C, T, V = x.size()

        x_input = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x_input = self.data_bn(x_input)
        x_input = x_input.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        if return_losses:
            phase_assignments, boundary_positions, boundary_scores = self.soft_decomp(x_input, return_scores=True)
        else:
            phase_assignments, boundary_positions = self.soft_decomp(x_input)
            boundary_scores = None

        x_class = x_input
        for i, layer in enumerate(self.class_backbone):
            x_class = layer(x_class, self._get_weighted_adjacency(self.class_edge_importance[i]))

        x_class = self.temporal_attention(x_class)

        phase_representations = []
        for p in range(self.num_phases):
            weights  = phase_assignments[:, p:p+1, :].unsqueeze(3)
            pooled   = (x_class * weights).sum(dim=[2, 3]) / (weights.sum(dim=[2, 3]) + 1e-8)
            pooled   = self.phase_transforms[p](pooled)
            phase_representations.append(pooled)

        stacked_phases = torch.stack(phase_representations, dim=1)

        if return_analysis:
            bag_feature, phase_attention = self.piw(stacked_phases, return_attention=True)
        else:
            bag_feature = self.piw(stacked_phases)

        logits  = self.classifier(bag_feature)
        outputs = [logits]

        if return_analysis:
            analysis = {
                'phase_assignments':  phase_assignments,
                'phase_attention':    phase_attention,
                'boundary_positions': boundary_positions,
            }
            outputs.append(analysis)

        if return_losses:
            boundary_sparsity, boundary_smoothness = self.soft_decomp.compute_boundary_loss(boundary_scores)
            losses = {
                'boundary_sparsity':   boundary_sparsity,
                'boundary_smoothness': boundary_smoothness,
                'adj_sparsity':        self.adaptive_adj.compute_sparsity_loss(),
            }
            outputs.append(losses)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)
