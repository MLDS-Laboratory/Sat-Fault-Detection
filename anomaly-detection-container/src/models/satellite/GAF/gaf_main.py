import os, argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim

from pipelines.esa_dataloader import ESAMissionDataLoader
from models.satellite.GAF.gaf_data_loader import GAFDataset, stratified_sample, mil_collate
from models.satellite.GAF.CNNs.pretrained_resnet import get_pretrained_resnet
from models.satellite.GAF.CNNs.scratch_cnn import CNNFromScratch
from models.satellite.GAF.CNNs.cnn_training import ModelTrainer
from utils.losses import CompoundLoss
from utils.env_utils import data_dir
from utils.wandb_utils import maybe_init_wandb, log_best_model_as_artifact

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(__file__, "../../../../data/ESA-Anomaly/ESA-Mission1")))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--model", choices=["pretrained","scratch"], default="pretrained")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--unfreeze_stem", action="store_true", default=True)
    return p.parse_args()

def run_main(model_name, model, hyperparams, mission_dir):
    # Load ESA segments with Event-aware splitting (Fix 1)
    loader = ESAMissionDataLoader(mission_dir=mission_dir)
    train_segs, val_segs, test_segs = loader.get_train_val_test_segments()

    # Down/select with Increased Budget (Fix 2)
    # Increase n from 100k/20k to 300k/60k
    # Since stratified_sample takes two lists, we process them carefully.
    # We can treat val_segs as part of the 'test' budget for stratified_sample if we combine them,
    # or just call it on each.
    
    def downsample_split(segs, max_samples, name):
        if len(segs) <= max_samples: return segs
        # stratified_sample is built for two lists, but we can pass an empty one
        out, _ = stratified_sample(segs, [], max_samples, 0, oversample_anomaly=False)
        print(f"Downsampled {name} to {len(out)} samples")
        return out

    train_segs = downsample_split(train_segs, 300000, "train")
    val_segs   = downsample_split(val_segs,   60000,  "val")
    test_segs  = downsample_split(test_segs,  60000,  "test")

    # grayscale transforms
    tfms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
    }
    train_ds = GAFDataset(train_segs, transform=tfms['train'], cache_dir="/tmp/gaf_cache")
    val_ds   = GAFDataset(val_segs,   transform=tfms['val'],   cache_dir="/tmp/gaf_cache")
    test_ds  = GAFDataset(test_segs,  transform=tfms['val'],   cache_dir="/tmp/gaf_cache")

    bs = hyperparams['batch_size']
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=False, prefetch_factor=1, collate_fn=mil_collate),
        'val':   DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=False, prefetch_factor=1, collate_fn=mil_collate),
        'test':  DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4, pin_memory=False, prefetch_factor=1, collate_fn=mil_collate),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = hyperparams['loss_fn']
    
    # Optimizer with Differential Learning Rate (Fix 6)
    if model_name == "pretrained" and hyperparams.get('unfreeze_stem', False):
        stem_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'conv1' in name or 'bn1' in name:
                stem_params.append(param)
            else:
                head_params.append(param)
        optimizer = optim.Adam([
            {'params': stem_params, 'lr': hyperparams['lr'] / 10.0},
            {'params': head_params, 'lr': hyperparams['lr']}
        ])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparams['lr'])
    # OneCycleLR Scheduler (Fix 7)
    acc_steps = 64
    total_optimizer_steps = hyperparams['epochs'] * ((len(dataloaders['train']) + acc_steps - 1) // acc_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=hyperparams['lr'], total_steps=total_optimizer_steps,
        pct_start=0.1, anneal_strategy='cos'
    )

    # W&B init
    run = maybe_init_wandb(project="gaf-anomaly-clf", config={
        "arch": model_name, "epochs": hyperparams['epochs'], "batch_size": bs, "lr": hyperparams['lr'],
        "unfreeze_stem": hyperparams.get('unfreeze_stem', False)
    })

    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device,
                           mixed_precision=hyperparams.get('mixed_precision', False),
                           wandb_run=run, scheduler=scheduler)

    model_trained, history = trainer.train(num_epochs=hyperparams['epochs'], accumulation_steps=acc_steps)
    test_acc, test_f05, test_cm = trainer.evaluate(phase='test')

    from utils.env_utils import model_dir
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

    if args.model == "scratch":
        model = CNNFromScratch(num_classes=2, input_size=224)
        model_name="scratch"
    else:
        model = get_pretrained_resnet(num_classes=2, freeze_early=True, unfreeze_stem=args.unfreeze_stem)
        model_name="pretrained"

    hp = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mixed_precision=args.mixed_precision,
        loss_fn=CompoundLoss(focal_weight=0.3, f05_weight=0.7), loss_name="CompoundLoss",
        unfreeze_stem=args.unfreeze_stem
    )
    res = run_main(model_name, model, hp, mission_dir=mission_dir)
    print(res)
