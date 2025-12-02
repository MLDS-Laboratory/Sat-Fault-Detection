import os, sys, json, csv
import pandas as pd
import torch

# ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.satellite.GAF.gaf_main import run_main
from models.satellite.GAF.CNNs.pretrained_resnet import get_pretrained_resnet
from models.satellite.GAF.CNNs.scratch_cnn import CNNFromScratch

def grid_search(output_csv="grid_search_results.csv"):
    # Define hyperparameter grid
    epochs_list       = [10]
    batch_sizes       = [32]
    learning_rates    = [5e-3]
    loss_fns = [
        # (torch.nn.CrossEntropyLoss(), "CrossEntropy"),
        (torch.nn.CrossEntropyLoss(label_smoothing=0.1), "CrossEntropy_ls0.1")
    ]

    # Create outputs folder relative to the directory this file is in
    base_dir = os.path.abspath(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    output_csv = os.path.join(outputs_dir, output_csv)

    # Define CSV header columns
    header = ['model', 'epochs', 'batch_size', 'lr', 'loss_fn',
              'test_acc', 'test_f1', 'test_cm',
              'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1']

    # Open the output file in write mode, and write the header
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

    # Also keep results in memory if desired 
    all_results = []

    # open file in append mode for incremental writing
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        
        for model_type in ['scratch', 'pretrained']:
            for epochs in epochs_list:
                for bs in batch_sizes:
                    for lr in learning_rates:
                        for loss_fn, loss_name in loss_fns:
                            # instantiate model
                            if model_type == 'scratch':
                                model = CNNFromScratch(num_classes=2, input_size=224)
                            else:
                                model = get_pretrained_resnet(num_classes=2, freeze_early=True)
                            hyperparams = {
                                'epochs': epochs,
                                'batch_size': bs,
                                'lr': lr,
                                'loss_fn': loss_fn,
                                'loss_name': loss_name
                            }
                            print(f"\n=== Running {model_type} | epochs={epochs}, bs={bs}, lr={lr}, loss={loss_name} ===")
                            res = run_main(model_type, model, hyperparams)
                            all_results.append(res)
                            
                            # Prepare row for CSV. Note: Convert the confusion matrix to JSON.
                            row = {
                                'model': res['model'],
                                'epochs': res['epochs'],
                                'batch_size': res['batch_size'],
                                'lr': res['lr'],
                                'loss_fn': res['loss_fn'],
                                'test_acc': res['test_acc'],
                                'test_f1': res['test_f1'],
                                'test_cm': json.dumps(res['test_confusion_matrix']),
                                'train_loss': json.dumps(res['train_loss']),
                                'val_loss': json.dumps(res['val_loss']),
                                'train_acc': json.dumps(res['train_acc']),
                                'val_acc': json.dumps(res['val_acc']),
                                'train_f1': json.dumps(res['train_f1']),
                                'val_f1': json.dumps(res['val_f1'])
                            }
                            
                            writer.writerow(row)
                            f.flush()   # flush to ensure it is written immediately
                            
    print(f"\nGrid search complete! Results saved to {output_csv}")

if __name__ == "__main__":
    grid_search("esa_grid_search_results.csv")
