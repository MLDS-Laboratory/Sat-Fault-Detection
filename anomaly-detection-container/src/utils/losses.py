import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, alpha=0.95, smoothing=0.0):
        super(AsymmetricFocalLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits, targets):
        """
        logits: [B, 2]
        targets: [B]
        """
        p = F.softmax(logits, dim=1)
        p_anomaly = p[:, 1]
        
        # Apply label smoothing if specified
        if self.smoothing > 0:
            with torch.no_grad():
                targets_smooth = targets.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        else:
            targets_smooth = targets.float()

        # Add cost-sensitive weighting as a safety net
        with torch.no_grad():
            n_anomaly = torch.sum(targets == 1).float()
            n_nominal = torch.sum(targets == 0).float()
            if n_anomaly > 0:
                cost_weight = n_nominal / n_anomaly
            else:
                cost_weight = 1.0

        # Compute weights for focal loss
        loss_pos = -self.alpha * (1 - p_anomaly)**self.gamma_pos * targets_smooth * torch.log(p_anomaly + 1e-8)
        loss_neg = -(1 - self.alpha) * p_anomaly**self.gamma_neg * (1 - targets_smooth) * torch.log(1 - p_anomaly + 1e-8)
        
        loss_pos = loss_pos * cost_weight
        
        return (loss_pos + loss_neg).mean()

class SoftF05Loss(nn.Module):
    def __init__(self, beta=0.5, epsilon=1e-8, min_positives=5):
        super(SoftF05Loss, self).__init__()
        self.beta = beta
        self.beta2 = beta ** 2
        self.epsilon = epsilon
        self.min_positives = min_positives
        self.p_buffer = []
        self.targets_buffer = []

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        p_anomaly = p[:, 1]
        
        # Fix 1 — Switch to a batch accumulation strategy
        self.p_buffer.append(p_anomaly)
        self.targets_buffer.append(targets.float())
        
        all_targets = torch.cat(self.targets_buffer)
        if all_targets.sum() < self.min_positives:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
            
        all_p = torch.cat(self.p_buffer)
        
        soft_TP = torch.sum(all_p * all_targets)
        soft_FP = torch.sum(all_p * (1 - all_targets))
        soft_FN = torch.sum((1 - all_p) * all_targets)

        soft_F05 = (1 + self.beta2) * soft_TP / ((1 + self.beta2) * soft_TP + self.beta2 * soft_FN + soft_FP + self.epsilon)
        
        return 1 - soft_F05

    def reset_buffer(self):
        self.p_buffer = []
        self.targets_buffer = []

class CompoundLoss(nn.Module):
    def __init__(self, focal_weight=0.5, f05_weight=0.5, gamma_pos=0.0, gamma_neg=4.0, alpha=0.95, beta=0.5, smoothing=0.01, min_positives=5):
        super(CompoundLoss, self).__init__()
        self.focal_weight = focal_weight
        self.f05_weight = f05_weight
        self.focal_loss = AsymmetricFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=alpha, smoothing=smoothing)
        self.soft_f05_loss = SoftF05Loss(beta=beta, min_positives=min_positives)

    def forward(self, logits, targets, use_f05=True):
        focal = self.focal_loss(logits, targets)
        if not use_f05:
            return focal
            
        soft_f05 = self.soft_f05_loss(logits, targets)
        return self.focal_weight * focal + self.f05_weight * soft_f05
    
    def reset_buffer(self):
        self.soft_f05_loss.reset_buffer()
