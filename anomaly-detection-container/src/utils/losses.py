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

        # Step 4 — Add cost-sensitive weighting as a safety net
        # compute cost_weight = (N_nominal / N_anomaly)
        with torch.no_grad():
            n_anomaly = torch.sum(targets == 1).float()
            n_nominal = torch.sum(targets == 0).float()
            if n_anomaly > 0:
                cost_weight = n_nominal / n_anomaly
            else:
                cost_weight = 1.0

        # Compute weights for focal loss
        # For targets == 1: (1 - p_anomaly)**gamma_pos
        # For targets == 0: p_anomaly**gamma_neg
        
        loss_pos = -self.alpha * (1 - p_anomaly)**self.gamma_pos * targets_smooth * torch.log(p_anomaly + 1e-8)
        loss_neg = -(1 - self.alpha) * p_anomaly**self.gamma_neg * (1 - targets_smooth) * torch.log(1 - p_anomaly + 1e-8)
        
        # Apply dynamic cost weighting to anomaly class
        # We only apply it to the positive loss component
        loss_pos = loss_pos * cost_weight
        
        return (loss_pos + loss_neg).mean()

class SoftF05Loss(nn.Module):
    def __init__(self, beta=0.5, epsilon=1e-8):
        super(SoftF05Loss, self).__init__()
        self.beta = beta
        self.beta2 = beta ** 2
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if targets.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
            
        p = F.softmax(logits, dim=1)
        p_anomaly = p[:, 1]
        targets = targets.float()

        soft_TP = torch.sum(p_anomaly * targets)
        soft_FP = torch.sum(p_anomaly * (1 - targets))
        soft_FN = torch.sum((1 - p_anomaly) * targets)

        soft_F05 = (1 + self.beta2) * soft_TP / ((1 + self.beta2) * soft_TP + self.beta2 * soft_FN + soft_FP + self.epsilon)
        
        return 1 - soft_F05

class CompoundLoss(nn.Module):
    def __init__(self, focal_weight=0.5, f05_weight=0.5, gamma_pos=0.0, gamma_neg=4.0, alpha=0.95, beta=0.5, smoothing=0.01):
        super(CompoundLoss, self).__init__()
        self.focal_weight = focal_weight
        self.f05_weight = f05_weight
        self.focal_loss = AsymmetricFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=alpha, smoothing=smoothing)
        self.soft_f05_loss = SoftF05Loss(beta=beta)

    def forward(self, logits, targets, use_f05=True):
        focal = self.focal_loss(logits, targets)
        if not use_f05:
            return focal
            
        soft_f05 = self.soft_f05_loss(logits, targets)
        return self.focal_weight * focal + self.f05_weight * soft_f05
