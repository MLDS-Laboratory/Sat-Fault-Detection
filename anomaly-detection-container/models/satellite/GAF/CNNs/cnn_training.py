import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import time
import os

class ModelTrainer:
    """
    Class to encapsulate training and evaluation routines for a given model.
    Records training and validation metrics per epoch for hyperparameter tuning.
    """
    def __init__(self, model, dataloaders, criterion, optimizer, device, mixed_precision=True):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = mixed_precision

        # Initialize history dict for tracking metrics
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss':   [], 'val_acc':   [], 'val_f1':   []
        }

    def save_model(self, path, save_entire_model=False):
        """
        Save model weights or the entire model to disk.
        
        Parameters:
        - path: Path where to save the model
        - save_entire_model: If True, saves the entire model. If False (default), saves only the state_dict
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
            
        if save_entire_model:
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


    def train(self, num_epochs=10):
        best_f1 = 0.0
        best_model_wts = None

        # mixed precision training
        scaler = torch.cuda.amp.GradScaler("cuda") if (self.device.type == 'cuda' and self.mixed_precision) else None
        
        # Track total training time
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for phase in ['train', 'val']:

                # choose the model mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                # track loss and accuracy
                running_loss = 0.0
                all_preds = []
                all_labels = []
                
                # Calculate total batches for progress bar
                total_batches = len(self.dataloaders[phase])
                
                # Create progress bar for this phase
                with tqdm(total=total_batches, desc=f'{phase.capitalize()}', ncols=100) as pbar:
                    batch_times = []
                    
                    # Start time for this batch
                    batch_start = time.time()
                    
                    # Loop through all batches with progress bar
                    for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        self.optimizer.zero_grad()
                        
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
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                        # Calculate time for this batch
                        batch_end = time.time()
                        batch_time = batch_end - batch_start
                        batch_times.append(batch_time)
                        batch_start = batch_end
                        
                        # Update progress bar with batch statistics
                        avg_batch_time = sum(batch_times) / len(batch_times)
                        eta = avg_batch_time * (total_batches - i - 1)
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'batch_time': f'{batch_time:.2f}s',
                            'ETA': f'{eta:.1f}s'
                        })
                        pbar.update(1)
                
                # Calculate epoch metrics
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = accuracy_score(all_labels, all_preds)
                epoch_f1 = f1_score(all_labels, all_preds, average='macro')
                
                # Store in history
                self.history[f'{phase}_loss'].append(epoch_loss)
                self.history[f'{phase}_acc'].append(epoch_acc)
                self.history[f'{phase}_f1'].append(epoch_f1)
                
                print(f"\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")
                
                # Keep track of best weights
                if phase == 'val' and epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = self.model.state_dict()

                    # save the best model
                    current_dir = os.path.dirname(os.path.abspath(__file__))

                    try:
                        self.save_model(os.path.join(current_dir, f"saved_models/{self.model.__class__.__name__}.pth"))
                    except Exception as e:
                        print(f"Error saving model to file: {e}")
        
        # Calculate total training time
        time_elapsed = time.time() - start_time
        print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        print(f"Best val F1: {best_f1:.4f}")
        
        # Load best model weights
        if best_model_wts:
            self.model.load_state_dict(best_model_wts)
        return self.model, self.history

    def evaluate(self, phase='test'):
        """
        Forward pass eval
        """

        self.model.eval()
        all_preds = []
        all_labels = []

        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # no grads to train eval
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
        return acc, f1, cm
