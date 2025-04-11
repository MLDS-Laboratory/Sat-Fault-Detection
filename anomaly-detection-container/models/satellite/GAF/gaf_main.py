import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import OpsSatDataLoader, OpsSatGAFDataset
from CNNs.pretrained_resnet import get_pretrained_resnet
from CNNs.scratch_cnn import CNNFromScratch
from CNNs.cnn_training import ModelTrainer
import torch.optim as optim
from torch.utils.data import Subset


def main():
    # file paths for the OPS-SAT dataset
    dataset_csv = "C:\\Users\\varun\\Documents\\UMD\\Research\\MLDS\\Sat-Fault-Detection\\anomaly-detection-container\\data\\OPS-SAT\\dataset.csv"
    segment_csv = "C:\\Users\\varun\\Documents\\UMD\\Research\\MLDS\\Sat-Fault-Detection\\anomaly-detection-container\\data\\OPS-SAT\\segments.csv"
    
    # load data using OpsSatDataLoader
    data_loader = OpsSatDataLoader(dataset_csv, segment_csv)
    train_segments, test_segments = data_loader.get_train_test_segments()
    
    # standard ImageNet normalization values (https://pytorch.org/vision/stable/models.html)
    # TODO: check if these values are appropriate for GAF images - recalc for custom CNN
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomHorizontalFlip(),  # TODO: add data augmentation if needed to generalize
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    
    # Create the OPS-SAT GAF dataset
    train_dataset_full = OpsSatGAFDataset(train_segments, transform=data_transforms['train'])
    test_dataset = OpsSatGAFDataset(test_segments, transform=data_transforms['val'])
    
    # split training dataset further into training and validation sets  - 80/20 split
    num_train = len(train_dataset_full)
    indices = list(range(num_train))
    split = int(0.8 * num_train)
    train_indices, val_indices = indices[:split], indices[split:]

    # create subsets     
    train_subset = Subset(train_dataset_full, train_indices)
    val_subset = Subset(train_dataset_full, val_indices)
    
    # TODO: change batch + hyperparameters
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2),
        'val': DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: mess with loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # scratch architecture
    model = CNNFromScratch(num_classes=2, input_size=224)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # pretrained resnet architecture
    # model = get_pretrained_resnet(num_classes=2, freeze_early=True)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # initialize the trainer and run training
    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device)
    trainer.train(num_epochs=10)
    print("Evaluating on test data:")
    trainer.evaluate(phase='test')

if __name__ == "__main__":
    main()
