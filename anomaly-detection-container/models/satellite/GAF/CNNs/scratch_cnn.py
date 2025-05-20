import torch.nn as nn


class CNNFromScratch(nn.Module):
    """
    A simple CNN architecture built from scratch.
    It consists of three convolutional layers followed by max pooling,
    then a fully connected classifier.
    """
    def __init__(self, num_classes=2, input_size=224):
        super(CNNFromScratch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 channels - standardized so same dataloader works for this + resnet
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Spatial dim reduction (idk but apparently good)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # compute final feature map size after 3 pooling layers:
        feature_map_size = input_size // 8
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * (feature_map_size ** 2), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten (makes it work idk)
        x = self.classifier(x)
        return x

