import os, argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim

from pipelines.esa_stacked_dataloader import ESAStackedDataLoader
from models.satellite.GAF.gaf_data_loader import mil_collate, GAFDataset, stratified_sample
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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--model", choices=["pretrained","scratch"], default="pretrained")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--local_weights", type=str, default=None, help="Local path to weights for transfer learning")
    p.add_argument("--unfreeze_stem", action="store_true", default=True)
    return p.parse_args()

def run_main(model_name, model, hyperparams, mission_dir):
    # Load ESA Stacked segments with Event-aware splitting (Fix 1)
    loader = ESAStackedDataLoader(mission_dir=mission_dir, nominal_segment_len=2048)
    train_segs, val_segs, test_segs = loader.get_train_val_test_segments()

    # DYNAMIC DOWNSAMPLING with Increased Budget (Fix 2)
    in_channels = train_segs[0]['ts'][0].shape[1] if train_segs[0]['ts'] else 1
    base_train_max = 300000 # Increased 3x from 100k
    base_test_max = 60000   # Increased 3x from 20k
    
    actual_train_max = base_train_max // in_channels
    actual_test_max = base_test_max // in_channels
    print(f"Adjusting max segments for {in_channels} channels: Train limit={actual_train_max}, Test/Val limit={actual_test_max}")

    def downsample_split(segs, max_samples, name):
        if len(segs) <= max_samples: return segs
        # stratified_sample from gaf_data_loader works for stacked too
        out, _ = stratified_sample(segs, [], max_samples, 0, oversample_anomaly=False)
        print(f"Downsampled {name} to {len(out)} samples")
        return out

    train_segs = downsample_split(train_segs, actual_train_max, "train")
    val_segs   = downsample_split(val_segs,   actual_test_max,  "val")
    test_segs  = downsample_split(test_segs,  actual_test_max,  "test")

    # Use GAFDataset which now handles MIL bags
    train_ds = GAFDataset(train_segs, cache_dir="/tmp/stacked_gaf_cache")
    val_ds   = GAFDataset(val_segs,   cache_dir="/tmp/stacked_gaf_cache")
    test_ds  = GAFDataset(test_segs,  cache_dir="/tmp/stacked_gaf_cache")

    bs = hyperparams['batch_size']
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=False, prefetch_factor=1, collate_fn=mil_collate),
        'val':   DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=False, prefetch_factor=1, collate_fn=mil_collate),
        'test':  DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4, pin_memory=False, prefetch_factor=1, collate_fn=mil_collate),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = hyperparams['loss_fn']
    
    # Optimizer with Differential Learning Rate (Fix 6)
    if "pretrained" in model_name and hyperparams.get('unfreeze_stem', False):
        stem_params = []
        head_params = []
        # In StackedResNet, the model is in self.resnet
        target_model = model.resnet if hasattr(model, 'resnet') else model
        for name, param in target_model.named_parameters():
            if not param.requires_grad: continue
            if 'conv1' in name or 'bn1' in name:
                stem_params.append(param)
            else:
                head_params.append(param)
        
        # If StackedResNet, also include the projection 1x1 conv in head_params
        if hasattr(model, 'proj'):
            for param in model.proj.parameters():
                if param.requires_grad: head_params.append(param)
        
        optimizer = optim.Adam([
            {'params': stem_params, 'lr': hyperparams['lr'] / 10.0},
            {'params': head_params, 'lr': hyperparams['lr']}
        ])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparams['lr'])
    
    # OneCycleLR Scheduler (Fix 7)
    total_steps = hyperparams['epochs'] * len(dataloaders['train'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=hyperparams['lr'], total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )

    run = maybe_init_wandb(project="gaf-anomaly-clf", config={
        "arch": model_name, "epochs": hyperparams['epochs'], "batch_size": bs, "lr": hyperparams['lr'], "type": "stacked",
        "unfreeze_stem": hyperparams.get('unfreeze_stem', False)
    })

    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device,
                           mixed_precision=hyperparams.get('mixed_precision', False),
                           wandb_run=run, scheduler=scheduler)

    model_trained, history = trainer.train(num_epochs=hyperparams['epochs'], accumulation_steps=64)
    test_acc, test_f05, test_cm = trainer.evaluate(phase='test')

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
        'test_f05': test_f05,
        'test_confusion_matrix': test_cm.tolist()
    }

if __name__ == "__main__":
    args = parse_args()
    mission_dir = data_dir(args.data_dir)

    import pandas as pd
    channels_csv_path = os.path.join(mission_dir, "channels.csv")
    if not os.path.exists(channels_csv_path):
        raise FileNotFoundError(f"Could not find {channels_csv_path} to determine channels.")
    
    in_channels = len(pd.read_csv(channels_csv_path))

    weights_dir = os.environ.get("SM_CHANNEL_WEIGHTS")
    transfer_path = None
    if weights_dir and os.path.exists(weights_dir):
        pth_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
        if pth_files: 
            transfer_path = os.path.join(weights_dir, pth_files[0])
    elif args.local_weights and os.path.exists(args.local_weights):
        transfer_path = args.local_weights

    if transfer_path:
        model = load_transfer_model(args.model, transfer_path, in_channels, num_classes=2, unfreeze_stem=args.unfreeze_stem)
        model_name = f"{args.model}_transfer"
    elif args.model == "scratch":
        model = StackedScratchCNN(in_channels=in_channels, num_classes=2)
        model_name = "scratch_stacked"
    else:
        model = StackedResNet(in_channels=in_channels, num_classes=2, freeze_early=True, unfreeze_stem=args.unfreeze_stem)
        model_name = "pretrained_stacked"

    hp = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mixed_precision=args.mixed_precision,
        loss_fn=CompoundLoss(focal_weight=0.5, f05_weight=0.5), loss_name="CompoundLoss",
        unfreeze_stem=args.unfreeze_stem
    )
    res = run_main(model_name, model, hp, mission_dir=mission_dir)
