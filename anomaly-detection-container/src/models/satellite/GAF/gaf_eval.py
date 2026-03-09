import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pipelines.esa_dataloader import ESAMissionDataLoader
from models.satellite.GAF.gaf_data_loader import GAFDataset
from models.satellite.GAF.CNNs.pretrained_resnet import get_pretrained_resnet
from models.satellite.GAF.CNNs.scratch_cnn import CNNFromScratch
from models.satellite.GAF.CNNs.cnn_training import ModelTrainer
from utils.env_utils import data_dir
from utils.wandb_utils import maybe_init_wandb

def parse_args():
    p = argparse.ArgumentParser()
    # SageMaker will map the "dataset" channel here
    p.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_DATASET", "data/ESA-Anomaly/ESA-Mission2"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model", choices=["pretrained", "scratch"], default="pretrained")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # 1. Locate the weights from the SageMaker "weights" channel
    weights_dir = os.environ.get("SM_CHANNEL_WEIGHTS")
    weights_path = None
    if weights_dir and os.path.exists(weights_dir):
        pth_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
        if pth_files:
            weights_path = os.path.join(weights_dir, pth_files[0])
            
    if not weights_path:
        raise FileNotFoundError("No .pth weights file found in the provided weights channel!")

    print(f"Loading weights from: {weights_path}")

    # 2. Instantiate the exact 1D architecture
    if args.model == "scratch":
        model = CNNFromScratch(in_channels=1, num_classes=2, input_size=224)
    else:
        model = get_pretrained_resnet(num_classes=2, freeze_early=True)

    # Load the state dict
    state_dict = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Load the target dataset (Zero-Shot domain)
    mission_dir = data_dir(args.data_dir)
    loader = ESAMissionDataLoader(mission_dir=mission_dir)
    _, test_segs = loader.get_train_test_segments()  # We only care about the test split for eval

    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 1D Grayscale normalization
    ])

    test_ds = GAFDataset(test_segs, transform=eval_tfms, cache_dir="/tmp/gaf_cache_eval")
    
    # We map this to the 'test' key so ModelTrainer.evaluate(phase='test') works perfectly
    dataloaders = {
        'test': DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
    }

    # 4. Initialize W&B and Trainer
    run = maybe_init_wandb(project="gaf-anomaly-clf", config={
        "arch": args.model, "batch_size": args.batch_size, "type": "zero_shot_eval"
    })

    # We can pass None for criterion and optimizer since evaluate() only does forward passes
    trainer = ModelTrainer(model, dataloaders, criterion=None, optimizer=None, device=device, wandb_run=run)

    print(f"Starting Zero-Shot Evaluation on {len(test_segs)} segments...")
    test_acc, test_f1, test_cm = trainer.evaluate(phase='test')

    if hasattr(run, "finish"):
        run.finish()

    print("Evaluation Complete!")

if __name__ == "__main__":
    main()