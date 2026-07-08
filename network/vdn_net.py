"""VDN 的无参数混合网络。"""

import torch.nn as nn
import torch


class VDNNet(nn.Module):
    """将所有智能体的局部 Q 值直接相加为联合 Q_tot。"""

    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, q_values):
        """沿智能体维求和，并保留该维以兼容训练目标形状。"""
        return torch.sum(q_values, dim=2, keepdim=True)

