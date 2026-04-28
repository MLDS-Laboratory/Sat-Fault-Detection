import os, argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim

from pipelines.esa_stacked_dataloader import ESAStackedDataLoader
from models.satellite.GAF.stacked_gaf_dataloader import StackedGAFDataset, stacked_stratified_sample
from models.satellite.GAF.CNNs.stacked_architectures import load_transfer_model, StackedResNet, StackedScratchCNN
from models.satellite.GAF.CNNs.cnn_training import ModelTrainer
from utils.losses import CompoundLoss
from utils.env_utils import data_dir, model_dir
from utils.wandb_utils import maybe_init_wandb, log_best_model_as_artifact

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(__file__, "../../../../data/ESA-Anomaly/ESA-Mission1")))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--model", choices=["pretrained","scratch"], default="pretrained")
    p.add_argument("--mixed_precision", action="store_true")
    # Added argument for local transfer weights testing
    p.add_argument("--local_weights", type=str, default=None, help="Local path to weights for transfer learning")
    return p.parse_args()

def run_main(model_name, model, hyperparams, mission_dir):
    # Load ESA Stacked segments
    loader = ESAStackedDataLoader(mission_dir=mission_dir, nominal_segment_len=2048)
    train_segs, test_segs = loader.get_train_test_segments()

    # DYNAMIC DOWNSAMPLING: Adjust limits based on channel count
    in_channels = train_segs[0]['ts'].shape[1] 
    base_train_max = 100000
    base_test_max = 20000
    
    actual_train_max = base_train_max // in_channels
    actual_test_max = base_test_max // in_channels
    print(f"Adjusting max segments for {in_channels} channels: Train limit={actual_train_max}, Test limit={actual_test_max}")

    train_segs, test_segs = stacked_stratified_sample(
        train_segs, test_segs, 
        max_train_samples=actual_train_max, 
        max_test_samples=actual_test_max
    )

    # Transforms (No ToTensor or Normalize here, as StackedGAFDataset handles tensor creation natively)
    full_train = StackedGAFDataset(train_segs, cache_dir="/tmp/stacked_gaf_cache")
    test_ds    = StackedGAFDataset(test_segs,  cache_dir="/tmp/stacked_gaf_cache")

    n = len(full_train); split = int(0.8*n)
    train_ds = Subset(full_train, list(range(split)))
    val_ds   = Subset(full_train, list(range(split, n)))

    bs = hyperparams['batch_size']
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=False, prefetch_factor=1),
        'val':   DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=False, prefetch_factor=1),
        'test':  DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4, pin_memory=False, prefetch_factor=1),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = hyperparams['loss_fn']
    
    # Only pass parameters that require gradients to the optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=hyperparams['lr'])

    # W&B init
    run = maybe_init_wandb(project="gaf-anomaly-clf", config={
        "arch": model_name, "epochs": hyperparams['epochs'], "batch_size": bs, "lr": hyperparams['lr'], "type": "stacked"
    })

    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device,
                           mixed_precision=hyperparams.get('mixed_precision', False),
                           wandb_run=run)

    model_trained, history = trainer.train(num_epochs=hyperparams['epochs'])
    test_acc, test_f1, test_cm = trainer.evaluate(phase='test')

    best_path = os.path.join(model_dir(), f"{model.__class__.__name__}_best.pth")
    if os.path.exists(best_path):
        log_best_model_as_artifact(best_path)

    if hasattr(run, "finish"):
        run.finish()

    return {
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

if __name__ == "__main__":
    args = parse_args()
    mission_dir = data_dir(args.data_dir)

    # 1. Peek at the data to dynamically get the number of channels
    import pandas as pd
    channels_csv_path = os.path.join(mission_dir, "channels.csv")
    if not os.path.exists(channels_csv_path):
        raise FileNotFoundError(f"Could not find {channels_csv_path} to determine channels.")
    
    in_channels = len(pd.read_csv(channels_csv_path))
    print(f"Detected {in_channels} channels for Stacked GAFs.")

    # 2. Check for Transfer Learning Weights (SageMaker mapped or Local)
    weights_dir = os.environ.get("SM_CHANNEL_WEIGHTS")
    transfer_path = None
    
    if weights_dir and os.path.exists(weights_dir):
        pth_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
        if pth_files: 
            transfer_path = os.path.join(weights_dir, pth_files[0])
    elif args.local_weights and os.path.exists(args.local_weights):
        transfer_path = args.local_weights

    # 3. Model Initialization
    if transfer_path:
        print(f"Initializing Transfer Learning from {transfer_path}")
        model = load_transfer_model(args.model, transfer_path, in_channels, num_classes=2)
        model_name = f"{args.model}_transfer"
    elif args.model == "scratch":
        print("Initializing Scratch CNN")
        model = StackedScratchCNN(in_channels=in_channels, num_classes=2)
        model_name = "scratch_stacked"
    else:
        print("Initializing Pretrained ResNet")
        model = StackedResNet(in_channels=in_channels, num_classes=2, freeze_early=True)
        model_name = "pretrained_stacked"

    hp = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mixed_precision=args.mixed_precision,
        loss_fn=CompoundLoss(focal_weight=0.7, f05_weight=0.3, gamma_neg=2.0, alpha=0.25), loss_name="CompoundLoss"
    )
    res = run_main(model_name, model, hp, mission_dir=mission_dir)