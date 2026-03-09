import torch
import torch.nn as nn
from torchvision import models

class StackedResNet(nn.Module):
    def __init__(self, in_channels, num_classes=2, freeze_early=True):
        super(StackedResNet, self).__init__()
        # The adaptive 1x1 projection block mapping N channels down to 3
        self.projection = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.resnet = models.resnet18(pretrained=True)
        
        if freeze_early:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.projection(x)
        return self.resnet(x)

class StackedScratchCNN(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(StackedScratchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),        # resnet uses this, so makes comparison fairer - helps normalize (free var is just residuals now)
            nn.ReLU(inplace=True),
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
        return self.classifier(self.features(x))

def load_transfer_model(model_type, weights_path, new_in_channels, num_classes=2):
    """
    Loads a Mission 1 model, swaps the input dimensionality layer, 
    and freezes the rest of the network for Mission 2 fine-tuning.
    """
    if model_type == "pretrained":
        model = StackedResNet(in_channels=new_in_channels, num_classes=num_classes, freeze_early=True)
        layer_to_ignore = 'projection'
    else:
        model = StackedScratchCNN(in_channels=new_in_channels, num_classes=num_classes)
        # Freeze everything initially
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze ONLY the new Conv1 layer and the classifier
        for param in model.features[0].parameters():
            param.requires_grad = True

        # "Best practice dictates that whenever you replace a feature-extraction layer, you must also leave the final 
        # classification head unfrozen. This allows the network to recalibrate its decision boundary to account for 
        # the slight shifts in the feature distributions coming from the new dataset."
        for param in model.classifier.parameters():
            param.requires_grad = True

        
        layer_to_ignore = 'features.0'

    # Load weights, ignoring the dimension-mismatched input layer
    state_dict = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' in state_dict: # Handle if saved as full checkpoint
        state_dict = state_dict['model_state_dict']
        
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith(layer_to_ignore)}
    model.load_state_dict(filtered_dict, strict=False)
    
    print(f"Successfully loaded {model_type} for transfer learning. Training layer: {layer_to_ignore}")
    return model