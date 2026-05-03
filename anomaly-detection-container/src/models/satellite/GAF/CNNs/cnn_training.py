import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, fbeta_score, precision_score, recall_score
from tqdm import tqdm
import time, os, traceback
import numpy as np
import wandb
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

    def compute_corrected_f05(self, y_true, y_pred_proba, event_ids, threshold=0.5):
        """
        Computes the ESA-ADB corrected F0.5 score at the event level.
        - Anomaly Events: Group samples with label=1 by event_id. 
          TP_e = unique event_id where any sample >= threshold.
          FN_e = unique event_id where no sample >= threshold.
        - Nominal: Treat each sample with label=0 as independent.
          FP_e = count of nominal samples misclassified as anomaly.
          TNR_t = correctly identified nominal samples / total nominal samples.
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 1. Handle Anomaly Events (label == 1)
        anomaly_mask = (y_true == 1)
        anom_event_ids = event_ids[anomaly_mask]
        anom_preds = y_pred[anomaly_mask]
        
        unique_anom_events = np.unique(anom_event_ids)
        tp_e = 0
        fn_e = 0
        
        for eid in unique_anom_events:
            # An event is a TP if ANY bag in the event is flagged
            event_preds = anom_preds[anom_event_ids == eid]
            if np.any(event_preds == 1):
                tp_e += 1
            else:
                fn_e += 1
        
        # 2. Handle Nominal Segments (label == 0)
        nominal_mask = (y_true == 0)
        nominal_preds = y_pred[nominal_mask]
        
        tn_t = np.sum(nominal_preds == 0)
        fp_e = np.sum(nominal_preds == 1)
        total_nominal = len(nominal_preds)
        
        tnr_t = tn_t / total_nominal if total_nominal > 0 else 1.0
        
        # 3. Corrected Metrics
        # Corrected Precision = (TP_e / (TP_e + FP_e)) * TNR_t
        precision_corr = (tp_e / (tp_e + fp_e)) * tnr_t if (tp_e + fp_e) > 0 else 0.0
        recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
        
        beta = 0.5
        beta2 = beta**2
        if (beta2 * precision_corr + recall_e) > 0:
            f05_corr = (1 + beta2) * (precision_corr * recall_e) / (beta2 * precision_corr + recall_e)
        else:
            f05_corr = 0.0
            
        return {
            'f05_corrected': f05_corr,
            'tp_e': tp_e,
            'fn_e': fn_e,
            'fp_e': fp_e,
            'tnr_t': tnr_t,
            'recall_e': recall_e,
            'precision_corr': precision_corr,
            'num_events': len(unique_anom_events)
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

    def train(self, num_epochs=10, accumulation_steps=1):
        best_smoothed_f05 = 0.0
        best_model_wts = None

        scaler = torch.cuda.amp.GradScaler("cuda") if (self.device.type == 'cuda' and self.mixed_precision) else None
        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"Epoch {epoch+1}/{num_epochs}")
            use_f05 = epoch >= 3
            if epoch == 3:
                print("Warm-up complete. Switching to full CompoundLoss (focal + soft-F0.5).")
            for phase in ['train', 'val']:
                self.model.train(phase == 'train')

                running_loss = 0.0
                all_preds, all_labels = [], []
                all_proba, all_event_ids = [], []
                total_batches = len(self.dataloaders[phase])

                epoch_total_norm = 0.0
                num_grad_steps = 0
                batches_processed = 0

                with tqdm(total=total_batches, desc=f'{phase.capitalize()}', ncols=100) as pbar:
                    t0 = time.time()
                    try:
                        for i, (bags, labels, event_ids) in enumerate(self.dataloaders[phase]):
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
                            all_event_ids.extend(event_ids.numpy())
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
                
                y_true = np.array(all_labels)
                y_prob = np.array(all_proba)
                y_event_ids = np.array(all_event_ids)
                
                p_anom = y_prob[y_true == 1]
                p_norm = y_prob[y_true == 0]
                mean_p_anom = p_anom.mean() if len(p_anom) > 0 else 0.0
                mean_p_norm = p_norm.mean() if len(p_norm) > 0 else 0.0
                sep_ratio = mean_p_anom / (mean_p_norm + 1e-8)
                pred_rate = (np.array(all_preds) == 1).mean()

                if phase == 'train':
                    # train metrics at 0.5
                    m = self.compute_corrected_f05(y_true, y_prob, y_event_ids, threshold=0.5)
                    train_recall_e = m['recall_e']
                    
                    if not use_f05:
                        if train_recall_e == 0.0:
                             print("Warning: Model has zero recall on training anomaly events.")
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
                    # Fix 1 — Quick threshold sweep on val using corrected F0.5
                    threshold_sweep = np.arange(0.5, 1.0, 0.05)
                    best_val_f05_corr = -1.0
                    best_val_t = 0.5
                    for t in threshold_sweep:
                        m = self.compute_corrected_f05(y_true, y_prob, y_event_ids, threshold=t)
                        if m['f05_corrected'] > best_val_f05_corr:
                            best_val_f05_corr = m['f05_corrected']
                            best_val_t = t
                    
                    metrics_at_05 = self.compute_corrected_f05(y_true, y_prob, y_event_ids, threshold=0.5)
                    
                    # Use threshold-tuned value as checkpoint criterion
                    self.smoothed_f05 = 0.6 * best_val_f05_corr + 0.4 * self.smoothed_f05 if epoch > 0 else best_val_f05_corr
                    
                    self.history['val_f05'].append(best_val_f05_corr)
                    
                    print(f"Val F0.5 (Corrected): {metrics_at_05['f05_corrected']:.4f} at 0.5, {best_val_f05_corr:.4f} at {best_val_t:.2f}")
                    log_dict.update({
                        f"{phase}/f05_corrected_at_05": metrics_at_05['f05_corrected'],
                        f"{phase}/f05_corrected_best": best_val_f05_corr,
                        f"{phase}/f05_best_threshold": best_val_t,
                        f"{phase}/f05_smoothed": self.smoothed_f05,
                        f"{phase}/precision_corr": metrics_at_05['precision_corr'],
                        f"{phase}/recall_e": metrics_at_05['recall_e'],
                        f"{phase}/tnr_t": metrics_at_05['tnr_t'],
                        f"{phase}/num_ground_truth_events": metrics_at_05['num_events']
                    })
                    
                    if self.wandb:
                        log_dict.update({
                            f"{phase}/probs_anomaly": wandb.Histogram(p_anom),
                            f"{phase}/probs_nominal": wandb.Histogram(p_norm)
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
        all_probs, all_labels, all_event_ids = [], [], []
        print(f"Pre-computing probabilities for threshold tuning...")

        with torch.no_grad():
            for loader_name, loader in [("train", train_loader), ("val", val_loader)]:
                for bags, labels, event_ids in tqdm(loader, desc=f"Extracting {loader_name} probs"):
                    bag_sizes = [b.shape[0] for b in bags]
                    flat_inputs = torch.cat(bags).to(self.device)
                    all_logits = self.model(flat_inputs)
                    bag_logits = torch.split(all_logits, bag_sizes)
                    for bl in bag_logits:
                        prob = torch.max(torch.softmax(bl, dim=1)[:, 1]).item()
                        all_probs.append(prob)
                    all_labels.extend(labels.numpy())
                    all_event_ids.extend(event_ids.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_event_ids = np.array(all_event_ids)

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        best_thresholds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(all_labels)), all_labels)):
            f_probs, f_labels, f_event_ids = all_probs[val_idx], all_labels[val_idx], all_event_ids[val_idx]
            thresholds = np.arange(0.05, 0.95, 0.01)
            f_best_f05, f_best_t = -1.0, 0.5
            for t in thresholds:
                res = self.compute_corrected_f05(f_labels, f_probs, f_event_ids, threshold=t)
                if res['f05_corrected'] > f_best_f05:
                    f_best_f05 = res['f05_corrected']; f_best_t = t
            best_thresholds.append(f_best_t)
            print(f"Fold {fold+1} Best Threshold: {f_best_t:.2f} (F0.5 Corr: {f_best_f05:.4f})")

        self.best_threshold = np.median(best_thresholds)
        print(f"Final CV Median Threshold: {self.best_threshold:.2f}")

        if hasattr(self.model, 'threshold') and isinstance(self.model.threshold, torch.Tensor):
            self.model.threshold.fill_(self.best_threshold)
            self.save_model(f"{self.model.__class__.__name__}_best.pth")
            print(f"Model re-saved with optimized threshold: {self.best_threshold:.2f}")

        if self.wandb:
            self.wandb.log({"tuning/cv_median_threshold": self.best_threshold})

    def tune_threshold(self, val_loader):
        pass

    def evaluate(self, phase='test'):
        self.model.eval()
        all_labels, all_proba, all_event_ids = [], [], []
        for bags, labels, event_ids in self.dataloaders[phase]:
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
                all_event_ids.extend(event_ids.numpy())

        y_true = np.array(all_labels)
        y_proba = np.array(all_proba)
        y_event_ids = np.array(all_event_ids)
        
        # DEBUG: Check event distribution
        print(f"\n[DEBUG {phase}] Total samples: {len(y_true)}")
        print(f"[DEBUG {phase}] Anomaly samples (y_true==1): {np.sum(y_true == 1)}")
        print(f"[DEBUG {phase}] Nominal samples (y_true==0): {np.sum(y_true == 0)}")
        print(f"[DEBUG {phase}] Unique event IDs overall: {len(np.unique(y_event_ids))}")
        print(f"[DEBUG {phase}] Unique event IDs in anomalies: {len(np.unique(y_event_ids[y_true == 1]))}")
        print(f"[DEBUG {phase}] Event ID range: [{np.min(y_event_ids)}, {np.max(y_event_ids)}]")
        unique_anom_ids = np.unique(y_event_ids[y_true == 1])
        anom_event_counts = {eid: np.sum(y_event_ids[y_true == 1] == eid) for eid in unique_anom_ids[:5]}
        print(f"[DEBUG {phase}] Sample counts per event (first 5): {anom_event_counts}")
        
        metrics = self.compute_corrected_f05(y_true, y_proba, y_event_ids, threshold=self.best_threshold)
        all_preds = (y_proba >= self.best_threshold).astype(int)

        acc = accuracy_score(y_true, all_preds)
        cm  = confusion_matrix(y_true, all_preds)
        
        print(f"\n{phase.capitalize()} Results (Threshold={self.best_threshold:.2f}):")
        print(f"Accuracy: {acc:.4f}")
        print(f"Corrected F0.5 (event-wise): {metrics['f05_corrected']:.4f}")
        print(f"Precision Corr (event-wise): {metrics['precision_corr']:.4f}")
        print(f"Recall (event-wise): {metrics['recall_e']:.4f}")
        print(f"TNR (temporal): {metrics['tnr_t']:.4f}")
        print(f"Ground Truth Events: {metrics['num_events']}")
        print(f"TP Events: {metrics['tp_e']}, FN Events: {metrics['fn_e']}")
        print("Confusion Matrix (at segment level):")
        print(cm)
        
        if self.wandb:
            import wandb
            self.wandb.log({
                f"{phase}/accuracy": acc,
                f"{phase}/f05_corrected": metrics['f05_corrected'],
                f"{phase}/precision_corr": metrics['precision_corr'],
                f"{phase}/recall_e": metrics['recall_e'],
                f"{phase}/tnr_t": metrics['tnr_t'],
                f"{phase}/num_events": metrics['num_events'],
                f"{phase}/tp_events": metrics['tp_e'],
                f"{phase}/fn_events": metrics['fn_e'],
                f"{phase}/confusion": wandb.plot.confusion_matrix(
                    y_true=y_true.tolist(), preds=all_preds.tolist(), class_names=["normal","anomaly"])
            })
        return acc, metrics['f05_corrected'], cm
