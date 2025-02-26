import os
import time
from tempfile import TemporaryDirectory

import torch


def train_model(fold, model, criterion, optimizer, num_epochs=25, patience=3, delta=0.5, title=None):
    """
    Trains a model using early stopping and saves the best model.

    :param fold: Current fold number (for cross-validation).
    :param model: PyTorch model to train.
    :param criterion: Loss function.
    :param optimizer: Optimization algorithm.
    :param num_epochs: Number of training epochs.
    :param patience: Patience for early stopping.
    :param delta: Minimum improvement threshold for early stopping.
    :param title: Model title for saving.
    :return: Trained model, best results, and training history.
    """
    start_time = time.time()
    best_acc = 0.0
    history = {'train': {'loss': [], 'accuracy': []}, 'valid': {'loss': [], 'accuracy': []}, 'epochs': num_epochs}
    best_res = None

    with TemporaryDirectory() as tempdir:
        best_model_path = os.path.join(tempdir, 'best_model_params.pt')
        early_stopper = EarlyStopper(patience, delta)

        for epoch in range(num_epochs):
            res = {'preds': [], 'labels': []}

            # Training phase
            model.train()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes['train']
            epoch_acc = running_corrects.double() / dataset_sizes['train']
            history['train']['loss'].append(epoch_loss)
            history['train']['accuracy'].append(epoch_acc.item())

            if epoch % 5 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}\n----------')
                print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Validation phase
            model.eval()
            running_loss, running_corrects = 0.0, 0

            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                    res['preds'].extend(preds.tolist())
                    res['labels'].extend(labels.tolist())

            epoch_loss = running_loss / dataset_sizes['valid']
            epoch_acc = running_corrects.double() / dataset_sizes['valid']
            history['valid']['loss'].append(epoch_loss)
            history['valid']['accuracy'].append(epoch_acc.item())

            if epoch % 5 == 0:
                print(f'Valid Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            # Save the best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                best_res = res

            # Check early stopping
            if early_stopper.early_stop(epoch_loss):
                print(f'Early stopping at epoch {epoch}')
                history['epochs'] = epoch
                break

        elapsed_time = time.time() - start_time
        print(f'Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
        print(f'Best validation accuracy: {best_acc:.4f}')

        # Save final model
        model_path = os.path.join(input_path, 'models', title)
        os.makedirs(model_path, exist_ok=True)
        model.load_state_dict(torch.load(best_model_path))
        torch.save(model.state_dict(), os.path.join(model_path, f'saved_model-{fold}.pth'))

    return model, best_res, history


def train_final_model(model, criterion, optimizer, num_epochs=25, title=None):
    """
    Trains the final model without validation and saves it.

    :param model: PyTorch model to train.
    :param criterion: Loss function.
    :param optimizer: Optimization algorithm.
    :param num_epochs: Number of training epochs.
    :param title: Model title for saving.
    :return: Trained model.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        if epoch % 5 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}\n----------')
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    model_path = os.path.join(input_path, 'models', title)
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'saved_model-final.pth'))
    return model


def evaluate(model, dataloader):
    """
    Evaluates the model on a given dataset.

    :param model: Trained PyTorch model.
    :param dataloader: Dataloader for evaluation.
    :return: Dictionary with predicted and true labels.
    """
    model.eval()
    res = {'preds': [], 'labels': []}
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            res['preds'].extend(preds.tolist())
            res['labels'].extend(labels.tolist())
    return res


def evaluate_img(model, input_img):
    """
    Evaluates a single image using the trained model.

    :param model: Trained PyTorch model.
    :param input_img: Input image tensor.
    :return: Predicted class.
    """
    model.eval()
    input_img = input_img.to(device)
    with torch.no_grad():
        output = model(input_img)
        _, pred = torch.max(output, 1)
    return pred
