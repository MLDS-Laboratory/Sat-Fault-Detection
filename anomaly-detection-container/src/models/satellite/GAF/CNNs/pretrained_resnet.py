from torchvision import models
import torch.nn as nn
import torch

class ResNet1DWrapper(nn.Module):
    """
    Wraps a pretrained ResNet to accept 1-channel (grayscale) GAF images
    by duplicating the single channel 3 times across the channel dimension.
    """
    def __init__(self, num_classes=2, freeze_early=True):
        super(ResNet1DWrapper, self).__init__()
        
        # Load the base model
        self.model = models.resnet18(pretrained=True)
        
        if freeze_early:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Replace final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        # x shape: [Batch, 1, Height, Width]
        # Repeat the 1 channel 3 times to create [Batch, 3, Height, Width]
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def get_pretrained_resnet(num_classes=2, freeze_early=True):
    """
    Load the wrapped pre-trained ResNet-18 model.
    """
    return ResNet1DWrapper(num_classes=num_classes, freeze_early=freeze_early)