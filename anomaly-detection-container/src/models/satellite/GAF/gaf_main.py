import os, sys, argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))

# from pipelines.ops_sat_dataloader import OpsSatDataLoader
from pipelines.esa_dataloader import ESAMissionDataLoader
from models.satellite.GAF.gaf_data_loader import GAFDataset, stratified_sample
from models.satellite.GAF.CNNs.pretrained_resnet import get_pretrained_resnet
from models.satellite.GAF.CNNs.scratch_cnn import CNNFromScratch
from models.satellite.GAF.CNNs.cnn_training import ModelTrainer
from utils.env_utils import data_dir
from utils.wandb_utils import maybe_init_wandb, log_best_model_as_artifact

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(__file__, "../../../../data/ESA-Anomaly/ESA-Mission1")))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--model", choices=["pretrained","scratch"], default="pretrained")
    p.add_argument("--mixed_precision", action="store_true")
    return p.parse_args()

def run_main(model_name, model, hyperparams, mission_dir):
    # Load ESA segments
    loader = ESAMissionDataLoader(mission_dir=mission_dir)
    train_segs, test_segs = loader.get_train_test_segments()

    # Down/select (you can tweak these caps)
    train_segs, test_segs = stratified_sample(train_segs, test_segs,
                                              max_train_samples=100000, max_test_samples=20000)

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
    full_train = GAFDataset(train_segs, transform=tfms['train'])
    test_ds    = GAFDataset(test_segs,  transform=tfms['val'])

    n = len(full_train); split = int(0.8*n)
    train_ds = Subset(full_train, list(range(split)))
    val_ds   = Subset(full_train, list(range(split, n)))

    bs = hyperparams['batch_size']
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=10, pin_memory=True),
        'val':   DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=10, pin_memory=True),
        'test':  DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=10, pin_memory=True),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = hyperparams['loss_fn']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparams['lr'])

    # W&B init (no-op if no WANDB_API_KEY)
    run = maybe_init_wandb(project="gaf-anomaly-clf", config={
        "arch": model_name, "epochs": hyperparams['epochs'], "batch_size": bs, "lr": hyperparams['lr']
    })

    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, device,
                           mixed_precision=hyperparams.get('mixed_precision', False),
                           wandb_run=run)

    model_trained, history = trainer.train(num_epochs=hyperparams['epochs'])
    test_acc, test_f1, test_cm = trainer.evaluate(phase='test')

    # Log the best model saved by trainer as a W&B artifact too
    # (filename uses class name *_best.pth)
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
        'test_f1': test_f1,
        'test_confusion_matrix': test_cm.tolist()
    }

if __name__ == "__main__":
    args = parse_args()
    mission_dir = data_dir(args.data_dir)  # respects SageMaker channel if present

    if args.model == "scratch":
        model = CNNFromScratch(num_classes=2, input_size=224); model_name="scratch"
    else:
        model = get_pretrained_resnet(num_classes=2, freeze_early=True); model_name="pretrained"

    hp = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mixed_precision=args.mixed_precision,
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.01), loss_name="CrossEntropy"
    )
    res = run_main(model_name, model, hp, mission_dir=mission_dir)
    print(res)
