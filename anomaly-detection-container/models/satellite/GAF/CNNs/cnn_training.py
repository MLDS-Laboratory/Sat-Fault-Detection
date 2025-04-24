import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class ModelTrainer:
    """
    Encapsulate training and evaluation routines for a given model.
    """
    def __init__(self, model, dataloaders, criterion, optimizer, device):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=10):
        best_f1 = 0.0
        best_model_wts = None

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
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # reset the gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':

                            # backprop and use the given optimizer (prolly Adam)
                            loss.backward()
                            self.optimizer.step()

                    # track the stats
                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(preds.cpu().numpy())            # move to cpu for numpy & scikit to use it
                    all_labels.extend(labels.cpu().numpy())

                # calculate epoch loss and accuracy
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = accuracy_score(all_labels, all_preds)
                epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

                if phase == 'val' and epoch_f1 > best_f1:
                    best_f1 = epoch_f1

                    # save the model weights if best
                    best_model_wts = self.model.state_dict()

        print("Training complete")
        if best_model_wts:
            self.model.load_state_dict(best_model_wts)
        return self.model

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
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        print(f"Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")
        print("Confusion Matrix:")                          
        print(cm)
        return acc, f1, cm
