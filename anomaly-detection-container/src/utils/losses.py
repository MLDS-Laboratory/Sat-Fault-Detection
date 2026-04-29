import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, alpha=0.85, smoothing=0.0):
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
    def __init__(self, beta=0.5, epsilon=1e-8, lambda_rec=0.1):
        super(SoftF05Loss, self).__init__()
        self.beta = beta
        self.beta2 = beta ** 2
        self.epsilon = epsilon
        self.lambda_rec = lambda_rec

    def forward(self, logits, targets):
        # Fix 1 — Remove accumulation buffer entirely
        if targets.sum() == 0:
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
            
        p = F.softmax(logits, dim=1)
        p_anomaly = p[:, 1]
        targets_f = targets.float()

        soft_TP = torch.sum(p_anomaly * targets_f)
        soft_FP = torch.sum(p_anomaly * (1 - targets_f))
        soft_FN = torch.sum((1 - p_anomaly) * targets_f)

        soft_F05 = (1 + self.beta2) * soft_TP / ((1 + self.beta2) * soft_TP + self.beta2 * soft_FN + soft_FP + self.epsilon)
        
        loss = 1 - soft_F05
        
        # Fix 5 — Add a recall floor regularization
        soft_recall = soft_TP / (soft_TP + soft_FN + self.epsilon)
        if soft_recall > 0.95:
            loss += self.lambda_rec * (soft_FP / (targets_f.size(0) - targets_f.sum() + self.epsilon))
            
        return loss

class CompoundLoss(nn.Module):
    def __init__(self, focal_weight=0.5, f05_weight=0.5, gamma_pos=0.0, gamma_neg=2.0, alpha=0.85, beta=0.5, smoothing=0.01):
        super(CompoundLoss, self).__init__()
        
        # Renormalize weights
        total_w = focal_weight + f05_weight
        self.focal_weight = focal_weight / total_w
        self.f05_weight = f05_weight / total_w
        
        self.focal_loss = AsymmetricFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=alpha, smoothing=smoothing)
        self.soft_f05_loss = SoftF05Loss(beta=beta)

    def forward(self, logits, targets, use_f05=True):
        focal = self.focal_loss(logits, targets)
        
        if not use_f05:
            return focal
            
        soft_f05 = self.soft_f05_loss(logits, targets)
        return self.focal_weight * focal + self.f05_weight * soft_f05
