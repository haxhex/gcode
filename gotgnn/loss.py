import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    from config import GoTGNNConfig
except ImportError:
    from ..config import GoTGNNConfig

class GoTMultiTaskLoss(nn.Module):

    def __init__(self, config: GoTGNNConfig):
        super().__init__()
        self.cfg = config
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_hat_node: torch.Tensor, y_true_node: torch.Tensor, node_label_mask: Optional[torch.Tensor], p_hat_pi: torch.Tensor, y_true_path: torch.Tensor, a_hat_type: torch.Tensor, a_true_type: torch.Tensor, a_hat_count: torch.Tensor, a_true_count: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l_node = self.mse_loss(y_hat_node, y_true_node) if y_hat_node.numel() > 0 else y_hat_node.sum() * 0.0
        if p_hat_pi is not None and p_hat_pi.numel() and y_true_path.numel():
            if p_hat_pi.shape != y_true_path.shape:
                raise ValueError('Predicted and target path scores must have matching shapes')
            l_path = self.mse_loss(p_hat_pi, y_true_path)
        else:
            l_path = y_hat_node.sum() * 0.0
        l_type = self.ce_loss(a_hat_type, a_true_type.reshape(-1))
        l_count = self.mse_loss(a_hat_count.reshape(-1), a_true_count.reshape(-1))
        total_loss = self.cfg.lambda_1 * l_node + self.cfg.lambda_2 * l_path + self.cfg.lambda_3 * l_type + self.cfg.lambda_4 * l_count
        return (total_loss, {'total_loss': float(total_loss.detach().item()), 'l_node_mse': float(l_node.detach().item()), 'l_path_mse': float(l_path.detach().item()), 'l_type_ce': float(l_type.detach().item()), 'l_count_mse': float(l_count.detach().item())})
