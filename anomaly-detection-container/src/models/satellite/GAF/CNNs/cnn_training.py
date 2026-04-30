import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, fbeta_score, precision_score, recall_score
from tqdm import tqdm
import time, os, traceback
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
        
        self.nominal_collapse_count = 0
        self.original_lr = self.optimizer.param_groups[0]['lr']
        self.smoothed_f05 = 0.0

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
        best_smoothed_f05 = 0.0
        best_model_wts = None

        # mixed precision training
        scaler = torch.cuda.amp.GradScaler("cuda") if (self.device.type == 'cuda' and self.mixed_precision) else None
        
        # Track total training time
        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"Epoch {epoch+1}/{num_epochs}")
            # Fix 4 — Extend warm-up to 3 epochs.
            use_f05 = epoch >= 3
            if epoch == 3:
                print("Warm-up complete. Switching to full CompoundLoss (focal + soft-F0.5).")

            for phase in ['train', 'val']:
                self.model.train(phase == 'train')

                running_loss = 0.0
                all_preds, all_labels = [], []
                all_proba = []
                total_batches = len(self.dataloaders[phase])
                
                epoch_total_norm = 0.0
                num_grad_steps = 0
                batches_processed = 0

                with tqdm(total=total_batches, desc=f'{phase.capitalize()}', ncols=100) as pbar:
                    t0 = time.time()
                    try:
                        for i, (bags, labels) in enumerate(self.dataloaders[phase]):
                            labels = labels.to(self.device)
                            bag_sizes = [b.shape[0] for b in bags]
                            inputs = torch.cat(bags).to(self.device)
                            
                            if phase == 'train' and i % accumulation_steps == 0:
                                self.optimizer.zero_grad(set_to_none=True)

                            with torch.set_grad_enabled(phase == 'train'):

                                if scaler is None:
                                    all_logits = self.model(inputs)
                                    bag_logits = torch.split(all_logits, bag_sizes)
                                    
                                    pooled_logits = []
                                    for bl in bag_logits:
                                        probs = torch.softmax(bl, dim=1)[:, 1]
                                        max_idx = torch.argmax(probs)
                                        pooled_logits.append(bl[max_idx])
                                    outputs = torch.stack(pooled_logits)
                                    
                                    proba = torch.softmax(outputs, dim=1)[:, 1]
                                    _, preds = torch.max(outputs, 1)
                                    
                                    if hasattr(self.criterion, 'forward') and 'use_f05' in self.criterion.forward.__code__.co_varnames:
                                        loss = self.criterion(outputs, labels, use_f05=use_f05)
                                    else:
                                        loss = self.criterion(outputs, labels)
                                    
                                    if phase == 'train':
                                        assert loss.requires_grad, f"Loss at epoch {epoch} batch {i} has no gradient."
                                        (loss / accumulation_steps).backward()
                                else:
                                    with torch.cuda.amp.autocast("cuda"):
                                        all_logits = self.model(inputs)
                                        bag_logits = torch.split(all_logits, bag_sizes)
                                        pooled_logits = []
                                        for bl in bag_logits:
                                            probs = torch.softmax(bl, dim=1)[:, 1]
                                            max_idx = torch.argmax(probs)
                                            pooled_logits.append(bl[max_idx])
                                        outputs = torch.stack(pooled_logits)
                                        
                                        proba = torch.softmax(outputs, dim=1)[:, 1]
                                        _, preds = torch.max(outputs, 1)
                                        if hasattr(self.criterion, 'forward') and 'use_f05' in self.criterion.forward.__code__.co_varnames:
                                            loss = self.criterion(outputs, labels, use_f05=use_f05)
                                        else:
                                            loss = self.criterion(outputs, labels)
                                    
                                    if phase == 'train':
                                        assert loss.requires_grad, f"Loss (AMP) at epoch {epoch} batch {i} has no gradient."
                                        scaler.scale(loss / accumulation_steps).backward()
                            
                            if phase == 'train':
                                if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
                                    if scaler is None:
                                        # Fix 5 — Gradient clipping at 1.0
                                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                        self.optimizer.step()
                                    else:
                                        scaler.unscale_(self.optimizer)
                                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                        scaler.step(self.optimizer)
                                        scaler.update()
                                    
                                    epoch_total_norm += total_norm.item()
                                    num_grad_steps += 1
                                    
                                    global_step += 1
                                    if self.wandb:
                                        log_msg = {"train/loss_step": float(loss.item()), "train/grad_norm": total_norm.item()}
                                        if not use_f05:
                                            log_msg["train/warmup_grad_norm"] = total_norm.item()
                                        self.wandb.log(log_msg)
                                        
                                    if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                                        self.scheduler.step()

                            running_loss += loss.item() * labels.size(0)
                            all_preds.extend(preds.detach().cpu().numpy())
                            all_labels.extend(labels.detach().cpu().numpy())
                            all_proba.extend(proba.detach().cpu().numpy())
                            batches_processed += 1

                            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                              'ETA': f'{(time.time()-t0)/(i+1)*(total_batches-i-1):.1f}s'})
                            pbar.update(1)
                    except Exception as e:
                        print(f"CRITICAL: Error in batch loop: {e}")
                        traceback.print_exc()
                        if self.wandb:
                            self.wandb.log({"error/traceback": traceback.format_exc()})
                        raise e

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc  = accuracy_score(all_labels, all_preds)
                epoch_f1   = f1_score(all_labels, all_preds, average='macro')
                
                # Fix 8 & 9 — Class-conditional probability logging
                y_true = np.array(all_labels)
                y_prob = np.array(all_proba)
                p_anom = y_prob[y_true == 1]
                p_norm = y_prob[y_true == 0]
                mean_p_anom = p_anom.mean() if len(p_anom) > 0 else 0.0
                mean_p_norm = p_norm.mean() if len(p_norm) > 0 else 0.0
                sep_ratio = mean_p_anom / (mean_p_norm + 1e-8)
                pred_rate = (np.array(all_preds) == 1).mean()

                if phase == 'train':
                    train_metrics = self.compute_sample_f05(y_true, y_prob, threshold=0.5)
                    train_recall = train_metrics['recall']
                    
                    if not use_f05:
                        if train_recall == 0.0:
                             print("Warning: Model has zero recall on training anomalies.")
                        if pred_rate < 0.001:
                            self.nominal_collapse_count += 1
                            if self.nominal_collapse_count >= 2:
                                print(f"CRITICAL: Nominal collapse detected (rate={pred_rate:.4f}). Shocking LR by 5.0x")
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = self.original_lr * 5.0
                                self.nominal_collapse_count = 0
                        else:
                            self.nominal_collapse_count = 0
                            if abs(self.optimizer.param_groups[0]['lr'] - self.original_lr) > 1e-9:
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = self.original_lr

                self.history[f'{phase}_loss'].append(epoch_loss)
                self.history[f'{phase}_acc'].append(epoch_acc)
                self.history[f'{phase}_f1'].append(epoch_f1)

                print(f"\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Macro-F1: {epoch_f1:.4f}")
                
                log_dict = {
                    f"{phase}/loss": epoch_loss,
                    f"{phase}/acc": epoch_acc,
                    f"{phase}/f1": epoch_f1,
                    f"{phase}/mean_p_anomaly_given_anomaly": mean_p_anom,
                    f"{phase}/mean_p_anomaly_given_nominal": mean_p_norm,
                    f"{phase}/separation_ratio": sep_ratio,
                    f"{phase}/anomaly_pred_rate": pred_rate,
                    "epoch": epoch
                }
                
                if phase == 'train' and num_grad_steps > 0:
                    log_dict["train/avg_grad_norm"] = epoch_total_norm / num_grad_steps

                if phase == 'val':
                    metrics = self.compute_sample_f05(y_true, y_prob, threshold=0.5)
                    epoch_f05 = metrics['f05']
                    # Fix 3 — Smooth val F0.5 for checkpointing (EMA)
                    self.smoothed_f05 = 0.6 * epoch_f05 + 0.4 * self.smoothed_f05 if epoch > 0 else epoch_f05
                    
                    self.history['val_f05'].append(epoch_f05)
                    self.history['val_precision'].append(metrics['precision'])
                    self.history['val_recall'].append(metrics['recall'])
                    self.history['val_tnr'].append(metrics['tnr'])
                    
                    print(f"Val F0.5: {epoch_f05:.4f} (Smoothed: {self.smoothed_f05:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f})")
                    log_dict.update({
                        f"{phase}/f05": epoch_f05,
                        f"{phase}/f05_smoothed": self.smoothed_f05,
                        f"{phase}/precision": metrics['precision'],
                        f"{phase}/recall": metrics['recall'],
                        f"{phase}/tnr": metrics['tnr']
                    })

                    if self.smoothed_f05 > best_smoothed_f05:
                        best_smoothed_f05 = self.smoothed_f05
                        best_model_wts = self.model.state_dict().copy()
                        self.save_model(f"{self.model.__class__.__name__}_best.pth")

                if self.wandb:
                    self.wandb.log(log_dict)

            epoch_end_time = time.time()
            if self.wandb:
                self.wandb.log({"epoch/duration": epoch_end_time - epoch_start_time, "epoch": epoch})

            if self.scheduler:
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                if self.wandb:
                    self.wandb.log({"train/lr": self.optimizer.param_groups[0]['lr'], "epoch": epoch})

        print(f"\nTraining complete. Best Smoothed Val F0.5: {best_smoothed_f05:.4f}")
        if best_model_wts:
            self.model.load_state_dict(best_model_wts)
        
        # After training, tune the threshold with K-fold CV
        print("Tuning decision threshold using 5-fold CV on train+val...")
        self.tune_threshold_cv(self.dataloaders['train'], self.dataloaders['val'])
        
        return self.model, self.history

    def tune_threshold_cv(self, train_loader, val_loader, n_folds=5):
        self.model.eval()
        
        def collect(loader):
            bags_all, labels_all = [], []
            for bags, lbls in loader:
                bags_all.extend(bags)
                labels_all.extend(lbls.numpy())
            return bags_all, labels_all
            
        t_bags, t_labels = collect(train_loader)
        v_bags, v_labels = collect(val_loader)
        
        all_samples = t_bags + v_bags
        all_labels = np.array(t_labels + v_labels)
        
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        print(f"Pre-computing probabilities for {len(all_samples)} samples...")
        all_probs = []
        with torch.no_grad():
            bs = train_loader.batch_size
            for i in range(0, len(all_samples), bs):
                batch_bags = all_samples[i : i+bs]
                bag_sizes = [b.shape[0] for b in batch_bags]
                flat_inputs = torch.cat(batch_bags).to(self.device)
                all_logits = self.model(flat_inputs)
                bag_logits = torch.split(all_logits, bag_sizes)
                for bl in bag_logits:
                    all_probs.append(torch.max(torch.softmax(bl, dim=1)[:, 1]).item())
        
        all_probs = np.array(all_probs)
        best_thresholds = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(all_labels)), all_labels)):
            f_probs = all_probs[val_idx]
            f_labels = all_labels[val_idx]
            
            thresholds = np.arange(0.05, 0.95, 0.01)
            f_best_f05 = -1; f_best_t = 0.5
            for t in thresholds:
                res = self.compute_sample_f05(f_labels, f_probs, threshold=t)
                if res['f05'] > f_best_f05:
                    f_best_f05 = res['f05']; f_best_t = t
            best_thresholds.append(f_best_t)
            print(f"Fold {fold+1} Best Threshold: {f_best_t:.2f} (F0.5: {f_best_f05:.4f})")
            
        self.best_threshold = np.median(best_thresholds)
        print(f"Final CV Median Threshold: {self.best_threshold:.2f}")
        if self.wandb:
            self.wandb.log({"tuning/cv_median_threshold": self.best_threshold})

    def tune_threshold(self, val_loader):
        pass

    def evaluate(self, phase='test'):
        """
        Forward pass eval
        """
        self.model.eval()
        all_labels, all_proba = [], []
        for bags, labels in self.dataloaders[phase]:
            bag_sizes = [b.shape[0] for b in bags]
            flat_inputs = torch.cat(bags).to(self.device)
            with torch.no_grad():
                all_logits = self.model(flat_inputs)
                bag_logits = torch.split(all_logits, bag_sizes)
                
                pooled_probs = []
                for bl in bag_logits:
                    probs = torch.softmax(bl, dim=1)[:, 1]
                    pooled_probs.append(torch.max(probs))
                
                all_proba.extend(torch.stack(pooled_probs).cpu().numpy())
                all_labels.extend(labels.numpy())

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
