import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import time, os
from utils.env_utils import model_dir, ensure_dir

class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, mixed_precision=False, wandb_run=None):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = mixed_precision
        self.wandb = wandb_run  # can be a no-op

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss':   [], 'val_acc':   [], 'val_f1':   []
        }

    def save_model(self, filename: str, save_entire_model=False):
        mdir = ensure_dir(model_dir())   # /opt/ml/model on SageMaker; ./outputs/model locally
        path = os.path.join(mdir, filename)
        if save_entire_model:
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        return path

    def train(self, num_epochs=10):
        best_f1 = 0.0
        best_model_wts = None

        # mixed precision training
        scaler = torch.cuda.amp.GradScaler("cuda") if (self.device.type == 'cuda' and self.mixed_precision) else None
        
        # Track total training time
        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for phase in ['train', 'val']:
                self.model.train(phase == 'train')

                running_loss = 0.0
                all_preds, all_labels = [], []
                total_batches = len(self.dataloaders[phase])

                with tqdm(total=total_batches, desc=f'{phase.capitalize()}', ncols=100) as pbar:
                    t0 = time.time()
                    for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer.zero_grad(set_to_none=True)

                        with torch.set_grad_enabled(phase == 'train'):

                            if scaler is None:
                                outputs = self.model(inputs)
                                _, preds = torch.max(outputs, 1)
                                loss = self.criterion(outputs, labels)
                                
                                if phase == 'train':
                                    loss.backward()
                                    self.optimizer.step()
                            else:
                                with torch.cuda.amp.autocast("cuda"):
                                    outputs = self.model(inputs)
                                    _, preds = torch.max(outputs, 1)
                                    loss = self.criterion(outputs, labels)
                                
                                if phase == 'train':
                                    scaler.scale(loss).backward()
                                    scaler.step(self.optimizer)
                                    scaler.update()
                        
                        # Update statistics
                        running_loss += loss.item() * inputs.size(0)
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())

                        if phase == 'train':
                            global_step += 1
                            self.wandb.log({"train/loss_step": float(loss.item())})

                        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                          'ETA': f'{(time.time()-t0)/(i+1)*(total_batches-i-1):.1f}s'})
                        pbar.update(1)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc  = accuracy_score(all_labels, all_preds)
                epoch_f1   = f1_score(all_labels, all_preds, average='macro')

                self.history[f'{phase}_loss'].append(epoch_loss)
                self.history[f'{phase}_acc'].append(epoch_acc)
                self.history[f'{phase}_f1'].append(epoch_f1)

                print(f"\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")
                self.wandb.log({f"{phase}/loss": epoch_loss,
                                f"{phase}/acc": epoch_acc,
                                f"{phase}/f1": epoch_f1,
                                "epoch": epoch})

                if phase == 'val' and epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = self.model.state_dict().copy()
                    # Save into model dir (SageMaker will auto-upload this dir to S3)
                    self.save_model(f"{self.model.__class__.__name__}_best.pth")

            # end phases
        print(f"\nTraining complete in {(time.time()-start_time)//60:.0f}m {(time.time()-start_time)%60:.0f}s")
        print(f"Best val F1: {best_f1:.4f}")

        if best_model_wts:
            self.model.load_state_dict(best_model_wts)
        return self.model, self.history

    def evaluate(self, phase='test'):
        """
        Forward pass eval
        """

        self.model.eval()
        all_preds, all_labels = [], []
        for inputs, labels in self.dataloaders[phase]:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            # track the stats by appending
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average='weighted')
        cm  = confusion_matrix(all_labels, all_preds)
        print(f"{phase.capitalize()} Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        if self.wandb:
            import wandb
            self.wandb.log({
                f"{phase}/accuracy": acc,
                f"{phase}/f1_weighted": f1,
                f"{phase}/confusion": wandb.plot.confusion_matrix(
                    y_true=all_labels, preds=all_preds, class_names=["normal","anomaly"])
            })
        return acc, f1, cm
