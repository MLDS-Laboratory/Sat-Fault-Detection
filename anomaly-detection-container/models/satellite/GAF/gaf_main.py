import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from CNNs.pretrained_resnet import get_pretrained_resnet
from CNNs.scratch_cnn import CNNFromScratch
from CNNs.cnn_training import ModelTrainer
import torch.optim as optim
from torch.utils.data import Subset

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))

# from pipelines.ops_sat_dataloader import OpsSatDataLoader
from pipelines.esa_dataloader import ESAMissionDataLoader
from models.satellite.GAF.gaf_data_loader import GAFDataset, stratified_sample


def run_main(model_name, model, hyperparams):
    """
    Run one training + eval with given model and hyperparameters.
    Returns a dict of:
      - 'model_name', 'epochs', 'batch_size', 'lr', 'loss_fn'
      - per-epoch training and validation metrics (lists)
      - final test accuracy, test F1, test confusion matrix
    """
    # Load data
    # dataset_csv = os.path.abspath(os.path.join(__file__, "../../../../data/OPS-SAT/dataset.csv"))
    # segment_csv = os.path.abspath(os.path.join(__file__, "../../../../data/OPS-SAT/segments.csv"))
    # loader = OpsSatDataLoader(dataset_csv, segment_csv)

    print("Create dataloaders...")
    mission_dir = os.path.abspath(os.path.join(__file__, "../../../../data/ESA-Anomaly/ESA-Mission1"))
    loader = ESAMissionDataLoader(mission_dir=mission_dir)

    print("Loading segments...")
    train_segs, test_segs = loader.get_train_test_segments()

    print(f"Original train segments: {len(train_segs)}, Original test segments: {len(test_segs)}")

    # Limit the number of samples in train and test sets - good for large datasets (like ESA)
    train_segs, test_segs = stratified_sample(train_segs, test_segs, max_train_samples=100000, max_test_samples=20000)

    # Print channel stats and distribution
    channel_stats = {}
    anomaly_count = 0
    total_count = len(train_segs)
    
    for seg in train_segs:  
        ch = seg['channel']
        if ch not in channel_stats:
            channel_stats[ch] = {'anomaly': 0, 'normal': 0, 'total': 0}
        
        if seg['label'] == 1:
            channel_stats[ch]['anomaly'] += 1
            anomaly_count += 1
        else:
            channel_stats[ch]['normal'] += 1
        channel_stats[ch]['total'] += 1

    print(f"Overall anomaly rate in training: {anomaly_count}/{total_count} ({anomaly_count/total_count*100:.2f}%)")
    print("Channel distribution in selected subset:")
    for ch, stats in sorted(channel_stats.items()):
        if stats['total'] > 0:  # Avoid division by zero
            print(f"Channel {ch}: {stats['total']} samples, {stats['anomaly']} anomalies ({stats['anomaly']/stats['total']*100:.2f}%)")

    print(f"Final counts: {len(train_segs)} train segments, {len(test_segs)} test segments")

    # Transforms
    tfms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

    # Datasets
    print("Transforming segments to GAF images...")
    full_train = GAFDataset(train_segs, transform=tfms['train'])
    test_ds    = GAFDataset(test_segs,  transform=tfms['val'])

    # Train/Val split
    n = len(full_train)
    split = int(0.8 * n)
    train_ds = Subset(full_train, list(range(split)))
    val_ds   = Subset(full_train, list(range(split, n)))

    # Dataloaders
    bs = hyperparams['batch_size']
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=10),
        'val':   DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=10),
        'test':  DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=10)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = hyperparams['loss_fn']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparams['lr'])

    # Trainer
    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device)

    # Train
    model_trained, history = trainer.train(num_epochs=hyperparams['epochs'])

    # Evaluate
    test_acc, test_f1, test_cm = trainer.evaluate(phase='test')

    # Aggregate results
    results = {
        'model': model_name,
        'epochs': hyperparams['epochs'],
        'batch_size': hyperparams['batch_size'],
        'lr': hyperparams['lr'],
        'loss_fn': hyperparams['loss_name'],
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'train_f1': history['train_f1'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'val_f1': history['val_f1'],
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_confusion_matrix': test_cm.tolist()
    }
    return results

if __name__ == "__main__":
    # Example: single experiment
    hp = dict(epochs=10, batch_size=32, lr=5e-3,
              loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.01), loss_name="CrossEntropy")
    # Model choice:
    # model = CNNFromScratch(num_classes=2, input_size=224); model_name="scratch"
    # or:
    model = get_pretrained_resnet(num_classes=2, freeze_early=True); model_name="pretrained"
    res = run_main(model_name, model, hp)
    print(res)