import logging
import os
import time
from tempfile import TemporaryDirectory
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, device: torch.device):
        """
        Initializes the Trainer class.

        :param model: PyTorch model to train.
        :param criterion: Loss function.
        :param optimizer: Optimization algorithm.
        :param device: Computation device (CPU/GPU).
        """
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer

        # Enable DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _run_epoch(self, dataloader: DataLoader, training: bool = True) -> Tuple[float, float]:
        """
        Runs a single training or validation epoch.

        :param dataloader: DataLoader for training or validation.
        :param training: If True, trains the model; otherwise, evaluates it.
        :return: Average loss and accuracy.
        """
        self.model.train() if training else self.model.eval()
        running_loss, correct_preds, total_samples = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                if training:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct_preds += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

        avg_loss = running_loss / total_samples
        avg_acc = correct_preds / total_samples
        return avg_loss, avg_acc

    def train(self, dataloaders: Dict[str, DataLoader], num_epochs: int = 25, patience: int = 3, delta: float = 0.5,
              save_path: str = "models", title: str = "model") -> Tuple[
        nn.Module, Dict, Dict]:
        """
        Trains the model using early stopping and saves the best model.

        :param dataloaders: Dictionary containing training & validation dataloaders.
        :param num_epochs: Number of training epochs.
        :param patience: Early stopping patience.
        :param delta: Minimum improvement threshold for early stopping.
        :param save_path: Directory to save trained models.
        :param title: Model title for saving.
        :return: Trained model, best results, and training history.
        """
        start_time = time.time()
        best_acc, best_res = 0.0, None
        history = {'train': {'loss': [], 'accuracy': []}, 'valid': {'loss': [], 'accuracy': []}, 'epochs': num_epochs}

        early_stopper = EarlyStopper(patience, delta)
        model_dir = os.path.join(save_path, title)
        os.makedirs(model_dir, exist_ok=True)

        with TemporaryDirectory() as tempdir:
            best_model_path = os.path.join(tempdir, 'best_model.pth')

            for epoch in range(num_epochs):
                train_loss, train_acc = self._run_epoch(dataloaders['train'], training=True)
                val_loss, val_acc = self._run_epoch(dataloaders['valid'], training=False)

                history['train']['loss'].append(train_loss)
                history['train']['accuracy'].append(train_acc)
                history['valid']['loss'].append(val_loss)
                history['valid']['accuracy'].append(val_acc)

                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Valid Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model.state_dict(), best_model_path)
                    best_res = {'accuracy': val_acc, 'loss': val_loss}

                # Early stopping check
                if early_stopper.early_stop(val_loss):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    history['epochs'] = epoch + 1
                    break

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s | Best Accuracy: {best_acc:.4f}")

        # Load and save the best model
        self.model.load_state_dict(torch.load(best_model_path))
        torch.save(self.model.state_dict(), os.path.join(model_dir, "best_model.pth"))

        return self.model, best_res, history

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates the model on a validation set.

        :param dataloader: Validation dataloader.
        :return: Average loss and accuracy.
        """
        return self._run_epoch(dataloader, training=False)

    def train_final(self, dataloader: DataLoader, num_epochs: int = 25, save_path: str = "models",
                    title: str = "final_model") -> nn.Module:
        """
        Trains the final model without validation and saves it.

        :param dataloader: Training dataloader.
        :param num_epochs: Number of training epochs.
        :param save_path: Directory to save trained models.
        :param title: Model title for saving.
        :return: Trained model.
        """
        for epoch in range(num_epochs):
            train_loss, train_acc = self._run_epoch(dataloader, training=True)
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        model_dir = os.path.join(save_path, title)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, "final_model.pth"))
        return self.model


class EarlyStopper:
    """
    Implements early stopping mechanism based on validation loss.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        """
        Determines whether to stop training early.

        :param validation_loss: Current validation loss.
        :return: True if training should stop early, else False.
        """
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
