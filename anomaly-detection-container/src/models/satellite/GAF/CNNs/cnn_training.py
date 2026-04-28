import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, fbeta_score, precision_score, recall_score
from tqdm import tqdm
import time, os
import numpy as np
from utils.env_utils import model_dir, ensure_dir

class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, mixed_precision=False, wandb_run=None, scheduler=None):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = mixed_precision
        self.wandb = wandb_run  # can be a no-op
        self.scheduler = scheduler
        self.best_threshold = 0.5

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss':   [], 'val_acc':   [], 'val_f1':   [],
            'val_f05':    [], 'val_precision': [], 'val_recall': [], 'val_tnr': []
        }

    def compute_sample_f05(self, y_true, y_pred_proba, threshold=0.5):
        """
        Computes sample-wise F0.5 approximation.
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            tnr = 0
            
        return {'f05': f05, 'precision': p, 'recall': r, 'tnr': tnr}

    def save_model(self, filename: str, save_entire_model=False):
        mdir = ensure_dir(model_dir())   # /opt/ml/model on SageMaker; ./outputs/model locally
        path = os.path.join(mdir, filename)
        if save_entire_model:
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        return path

    def train(self, num_epochs=10, accumulation_steps=1):
        best_f05 = 0.0
        best_model_wts = None

        # mixed precision training
        scaler = torch.cuda.amp.GradScaler("cuda") if (self.device.type == 'cuda' and self.mixed_precision) else None
        
        # Track total training time
        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            # Fix — Add a warm-up phase. First 2 epochs, focal only.
            use_f05 = epoch >= 2
            if epoch == 2:
                print("Warm-up complete. Switching to full CompoundLoss (focal + soft-F0.5).")

            for phase in ['train', 'val']:
                self.model.train(phase == 'train')

                running_loss = 0.0
                all_preds, all_labels = [], []
                all_proba = []
                total_batches = len(self.dataloaders[phase])
                
                epoch_total_norm = 0.0
                num_grad_steps = 0

                with tqdm(total=total_batches, desc=f'{phase.capitalize()}', ncols=100) as pbar:
                    t0 = time.time()
                    for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        if phase == 'train' and i % accumulation_steps == 0:
                            self.optimizer.zero_grad(set_to_none=True)

                        with torch.set_grad_enabled(phase == 'train'):

                            if scaler is None:
                                outputs = self.model(inputs)
                                proba = torch.softmax(outputs, dim=1)[:, 1]
                                _, preds = torch.max(outputs, 1)
                                
                                # Use use_f05 flag if criterion supports it
                                if hasattr(self.criterion, 'forward') and 'use_f05' in self.criterion.forward.__code__.co_varnames:
                                    loss = self.criterion(outputs, labels, use_f05=use_f05)
                                else:
                                    loss = self.criterion(outputs, labels)
                                
                                if phase == 'train':
                                    # Normalize loss for accumulation
                                    (loss / accumulation_steps).backward()
                            else:
                                with torch.cuda.amp.autocast("cuda"):
                                    outputs = self.model(inputs)
                                    proba = torch.softmax(outputs, dim=1)[:, 1]
                                    _, preds = torch.max(outputs, 1)
                                    if hasattr(self.criterion, 'forward') and 'use_f05' in self.criterion.forward.__code__.co_varnames:
                                        loss = self.criterion(outputs, labels, use_f05=use_f05)
                                    else:
                                        loss = self.criterion(outputs, labels)
                                
                                if phase == 'train':
                                    scaler.scale(loss / accumulation_steps).backward()
                        
                        if phase == 'train':
                            if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
                                # Add a gradient norm log
                                if scaler is None:
                                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                    self.optimizer.step()
                                else:
                                    scaler.unscale_(self.optimizer)
                                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                    scaler.step(self.optimizer)
                                    scaler.update()
                                
                                epoch_total_norm += total_norm.item()
                                num_grad_steps += 1
                                
                                # Reset buffer after gradient step
                                if hasattr(self.criterion, 'reset_buffer'):
                                    self.criterion.reset_buffer()
                                
                                global_step += 1
                                if self.wandb:
                                    self.wandb.log({"train/loss_step": float(loss.item()), "train/grad_norm": total_norm.item()})

                        # Update statistics
                        running_loss += loss.item() * inputs.size(0)
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())
                        all_proba.extend(proba.detach().cpu().numpy())

                        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                          'ETA': f'{(time.time()-t0)/(i+1)*(total_batches-i-1):.1f}s'})
                        pbar.update(1)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc  = accuracy_score(all_labels, all_preds)
                epoch_f1   = f1_score(all_labels, all_preds, average='macro')

                # Fix — Add a sanity check assertion for train_recall
                if phase == 'train':
                    train_metrics = self.compute_sample_f05(np.array(all_labels), np.array(all_proba), threshold=0.5)
                    train_recall = train_metrics['recall']
                    if epoch < 2 and train_recall == 0.0:
                         print("Warning: Model has zero recall on training anomalies. Focal loss weighting should correct this...")
                    if epoch == 1 and train_recall == 0.0:
                         print("CRITICAL: Model may be collapsed — check class weights and learning rate. train_recall is still 0.0")

                self.history[f'{phase}_loss'].append(epoch_loss)
                self.history[f'{phase}_acc'].append(epoch_acc)
                self.history[f'{phase}_f1'].append(epoch_f1)

                print(f"\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Macro-F1: {epoch_f1:.4f}")
                
                log_dict = {
                    f"{phase}/loss": epoch_loss,
                    f"{phase}/acc": epoch_acc,
                    f"{phase}/f1": epoch_f1,
                    "epoch": epoch
                }
                
                if phase == 'train' and num_grad_steps > 0:
                    log_dict["train/avg_grad_norm"] = epoch_total_norm / num_grad_steps

                if phase == 'val':
                    metrics = self.compute_sample_f05(np.array(all_labels), np.array(all_proba), threshold=0.5)
                    epoch_f05 = metrics['f05']
                    self.history['val_f05'].append(epoch_f05)
                    self.history['val_precision'].append(metrics['precision'])
                    self.history['val_recall'].append(metrics['recall'])
                    self.history['val_tnr'].append(metrics['tnr'])
                    
                    print(f"Val F0.5: {epoch_f05:.4f} (Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, TNR: {metrics['tnr']:.4f})")
                    log_dict.update({
                        f"{phase}/f05": epoch_f05,
                        f"{phase}/precision": metrics['precision'],
                        f"{phase}/recall": metrics['recall'],
                        f"{phase}/tnr": metrics['tnr']
                    })

                    if epoch_f05 > best_f05:
                        best_f05 = epoch_f05
                        best_model_wts = self.model.state_dict().copy()
                        self.save_model(f"{self.model.__class__.__name__}_best.pth")

                if self.wandb:
                    self.wandb.log(log_dict)

            # end phases
            # Fix 3 — Call scheduler.step() at the end of each epoch
            if self.scheduler:
                self.scheduler.step()
                if self.wandb:
                    self.wandb.log({"train/lr": self.optimizer.param_groups[0]['lr'], "epoch": epoch})

        print(f"\nTraining complete in {(time.time()-start_time)//60:.0f}m {(time.time()-start_time)%60:.0f}s")
        print(f"Best val F0.5: {best_f05:.4f}")

        if best_model_wts:
            self.model.load_state_dict(best_model_wts)
        
        # After training, tune the threshold
        print("Tuning decision threshold on validation set...")
        self.tune_threshold(self.dataloaders['val'])
        
        return self.model, self.history

    def tune_threshold(self, val_loader):
        self.model.eval()
        all_labels, all_proba = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                proba = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.numpy())
                all_proba.extend(proba.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_proba = np.array(all_proba)
        
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_f05 = -1
        best_thresh = 0.5
        
        for t in thresholds:
            res = self.compute_sample_f05(y_true, y_proba, threshold=t)
            if res['f05'] > best_f05:
                best_f05 = res['f05']
                best_thresh = t
        
        self.best_threshold = best_thresh
        final_metrics = self.compute_sample_f05(y_true, y_proba, threshold=self.best_threshold)
        
        print(f"Best Threshold: {self.best_threshold:.2f} -> Val F0.5: {best_f05:.4f}")
        if self.wandb:
            self.wandb.log({
                "tuning/best_threshold": self.best_threshold,
                "tuning/val_f05": best_f05,
                "tuning/val_precision": final_metrics['precision'],
                "tuning/val_recall": final_metrics['recall']
            })

    def evaluate(self, phase='test'):
        """
        Forward pass eval
        """
        self.model.eval()
        all_labels, all_proba = [], []
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                proba = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.numpy())
            all_proba.extend(proba.cpu().numpy())

        y_true = np.array(all_labels)
        y_proba = np.array(all_proba)
        
        metrics = self.compute_sample_f05(y_true, y_proba, threshold=self.best_threshold)
        all_preds = (y_proba >= self.best_threshold).astype(int)

        acc = accuracy_score(y_true, all_preds)
        cm  = confusion_matrix(y_true, all_preds)
        
        print(f"\n{phase.capitalize()} Results (Threshold={self.best_threshold:.2f}):")
        print(f"Accuracy: {acc:.4f}")
        print(f"F0.5 (sample-wise): {metrics['f05']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"TNR: {metrics['tnr']:.4f}")
        print("Note: These are sample-wise F0.5 approximations; use official ESA-ADB script for event-wise metrics.")
        print("Confusion Matrix:")
        print(cm)
        
        if self.wandb:
            import wandb
            self.wandb.log({
                f"{phase}/accuracy": acc,
                f"{phase}/f05_sample": metrics['f05'],
                f"{phase}/precision": metrics['precision'],
                f"{phase}/recall": metrics['recall'],
                f"{phase}/tnr": metrics['tnr'],
                f"{phase}/confusion": wandb.plot.confusion_matrix(
                    y_true=y_true.tolist(), preds=all_preds.tolist(), class_names=["normal","anomaly"])
            })
        return acc, metrics['f05'], cm

