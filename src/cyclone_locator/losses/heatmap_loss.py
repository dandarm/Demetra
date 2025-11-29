import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapMSE(nn.Module):
    """MSE sulla heatmap, calcolata su tutti i campioni (positivi e negativi)."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred, target: (B,1,H,W)
        if pred.shape != target.shape:
            raise ValueError("pred/target shape mismatch")
        return F.mse_loss(pred, target)
