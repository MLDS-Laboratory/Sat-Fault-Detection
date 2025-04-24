import os, sys, json
import pandas as pd
import torch

# ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from gaf_main import run_experiment
from CNNs.pretrained_resnet import get_pretrained_resnet
from CNNs.scratch_cnn import CNNFromScratch

def grid_search(output_csv="grid_search_results.csv"):
    # Define hyperparameter grid
    epochs_list       = [5, 10, 15]
    batch_sizes       = [16, 32, 64]
    learning_rates    = [1e-2, 1e-3, 1e-4]
    loss_fns = [
        (torch.nn.CrossEntropyLoss(), "CrossEntropy"),
        (torch.nn.CrossEntropyLoss(label_smoothing=0.1), "CrossEntropy_ls0.1"),
        (torch.nn.CrossEntropyLoss(label_smoothing=0.25), "CrossEntropy_ls0.25")
    ]

    all_results = []

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
                        res = run_experiment(model_type, model, hyperparams)
                        all_results.append(res)

    # Flatten results for CSV
    rows = []
    for r in all_results:
        row = {
            'model': r['model'],
            'epochs': r['epochs'],
            'batch_size': r['batch_size'],
            'lr': r['lr'],
            'loss_fn': r['loss_fn'],
            'test_acc': r['test_acc'],
            'test_f1': r['test_f1'],
            'test_cm': json.dumps(r['test_confusion_matrix'])
        }
        # you could also store per-epoch metrics as JSON strings if desired
        row['train_loss'] = json.dumps(r['train_loss'])
        row['val_loss']   = json.dumps(r['val_loss'])
        row['train_acc']  = json.dumps(r['train_acc'])
        row['val_acc']    = json.dumps(r['val_acc'])
        row['train_f1']   = json.dumps(r['train_f1'])
        row['val_f1']     = json.dumps(r['val_f1'])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nGrid search complete! Results saved to {output_csv}")

if __name__ == "__main__":
    grid_search()
