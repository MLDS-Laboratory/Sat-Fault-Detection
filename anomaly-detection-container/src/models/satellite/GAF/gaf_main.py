import os, argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim

from pipelines.esa_dataloader import ESAMissionDataLoader
from models.satellite.GAF.gaf_data_loader import GAFDataset, stratified_sample
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
    p.add_argument("--lr", type=float, default=2e-4) # Fix 2 — Increase LR back to 2e-4
    p.add_argument("--model", choices=["pretrained","scratch"], default="pretrained")
    p.add_argument("--mixed_precision", action="store_true")
    return p.parse_args()

def run_main(model_name, model, hyperparams, mission_dir):
    # Load ESA segments
    loader = ESAMissionDataLoader(mission_dir=mission_dir)
    train_segs, test_segs = loader.get_train_test_segments()

    # Down/select
    train_segs, test_segs = stratified_sample(train_segs, test_segs,
                                              max_train_samples=100000, max_test_samples=20000, 
                                              oversample_anomaly=False)

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
    full_train = GAFDataset(train_segs, transform=tfms['train'], cache_dir="/tmp/gaf_cache")
    test_ds    = GAFDataset(test_segs,  transform=tfms['val'], cache_dir="/tmp/gaf_cache")

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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparams['lr'])
    
    # Fix 3 — Add a cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

    # W&B init (no-op if no WANDB_API_KEY)
    run = maybe_init_wandb(project="gaf-anomaly-clf", config={
        "arch": model_name, "epochs": hyperparams['epochs'], "batch_size": bs, "lr": hyperparams['lr']
    })

    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device,
                           mixed_precision=hyperparams.get('mixed_precision', False),
                           wandb_run=run, scheduler=scheduler)

    # Using accumulation_steps=64 to ensure soft-F0.5 buffer hits min_positives=5 regularly
    model_trained, history = trainer.train(num_epochs=hyperparams['epochs'], accumulation_steps=64)
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
        model = CNNFromScratch(num_classes=2, input_size=224); model_name="scratch"
    else:
        model = get_pretrained_resnet(num_classes=2, freeze_early=True); model_name="pretrained"

    hp = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mixed_precision=args.mixed_precision,
        loss_fn=CompoundLoss(focal_weight=0.4, f05_weight=0.6, tnr_weight=0.3), loss_name="CompoundLoss"
    )
    res = run_main(model_name, model, hp, mission_dir=mission_dir)
    print(res)
