import torch.nn as nn
import torch


class CNNFromScratch(nn.Module):
    """
    A simple CNN architecture built from scratch.
    It consists of three convolutional layers followed by max pooling,
    then a fully connected classifier.
    """
    def __init__(self, in_channels=1, num_classes=2, input_size=224):
        super(CNNFromScratch, self).__init__()
        
        # Optimized decision threshold buffer
        self.register_buffer('threshold', torch.tensor(0.5))  

        self.features = nn.Sequential(
            # Takes in_channels (1 for regular GAFs, N for Stacked)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),        # resnet uses this, so makes comparison fairer - helps normalize (free var is just residuals now)
            nn.ReLU(inplace=True),
            # Ensures output is always 7x7 before the classifier
            nn.AdaptiveAvgPool2d((7, 7)) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x