from torchvision import models
import torch.nn as nn


def get_pretrained_resnet(num_classes=2, freeze_early=True):
    """
    Load a pre-trained ResNet-18 model and replace the final FC layer.
    freeze the early layers so only later layers are fine-tuned.
    """
    model = models.resnet18(pretrained=True)
    if freeze_early:
        for param in model.parameters():
            param.requires_grad = False

    # replace final fully connected layer to match the target classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model