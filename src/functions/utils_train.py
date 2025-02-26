import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from imblearn.over_sampling import RandomOverSampler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_training_results(history: Dict[str, Dict[str, list]], axis, fold: int) -> None:
    """
    Plots training and validation loss and accuracy.

    :param history: Dictionary containing loss and accuracy history.
    :param axis: Matplotlib axis for plotting.
    :param fold: Fold index for cross-validation.
    """
    epochs = range(len(history['train']['loss']))

    # Loss Plot
    axis[fold, 0].plot(epochs, history['train']['loss'], label="Train Loss")
    axis[fold, 0].plot(epochs, history['valid']['loss'], label="Valid Loss")
    axis[fold, 0].set_xlabel("Epochs")
    axis[fold, 0].set_ylabel("CrossEntropy Loss")
    axis[fold, 0].set_title("Loss Over Epochs")
    axis[fold, 0].grid(linewidth=1.5, alpha=0.25, linestyle="dashed")
    axis[fold, 0].legend()

    # Accuracy Plot
    axis[fold, 1].plot(epochs, history['train']['accuracy'], label="Train Accuracy")
    axis[fold, 1].plot(epochs, history['valid']['accuracy'], label="Valid Accuracy")
    axis[fold, 1].set_xlabel("Epochs")
    axis[fold, 1].set_ylabel("Accuracy")
    axis[fold, 1].set_ylim(0, 1)
    axis[fold, 1].set_title("Accuracy Over Epochs")
    axis[fold, 1].grid(linewidth=1.5, alpha=0.25, linestyle="dashed")
    axis[fold, 1].legend()


def denormalize(images: torch.Tensor, means: Tuple[float, float, float],
                stds: Tuple[float, float, float]) -> torch.Tensor:
    """
    Denormalizes a batch of images.

    :param images: Normalized image tensor.
    :param means: Tuple containing mean values for each channel.
    :param stds: Tuple containing standard deviation values for each channel.
    :return: Denormalized images.
    """
    means = torch.tensor(means).reshape(1, 3, 1, 1).to(images.device)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1).to(images.device)
    return images * stds + means


def display_images(images: torch.Tensor, labels: torch.Tensor, class_mapping: Dict[int, str],
                   num_images: int = 25) -> None:
    """
    Displays a batch of images with corresponding labels.

    :param images: Batch of images.
    :param labels: Corresponding labels.
    :param class_mapping: Dictionary mapping class indices to names.
    :param num_images: Number of images to display (default=25).
    """
    plt.figure(figsize=(20, 14))
    num_images = min(num_images, images.shape[0])

    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(class_mapping.get(labels[i].item(), "Unknown"))
        plt.axis("off")

    plt.show()


def oversample_data(index: torch.Tensor, y: torch.Tensor, seed: int = 0, sampling_strategy: float = 1.0) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Performs random oversampling to balance class distribution.

    :param index: Tensor of indices corresponding to dataset samples.
    :param y: Tensor of class labels.
    :param seed: Random seed for reproducibility.
    :param sampling_strategy: Oversampling ratio.
    :return: Resampled indices and corresponding labels.
    """
    if not isinstance(index, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Both 'index' and 'y' must be torch.Tensor")

    index_2d, y_2d = index.view(-1, 1).cpu().numpy(), y.view(-1, 1).cpu().numpy()
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
    index_resampled, y_resampled = ros.fit_resample(index_2d, y_2d)

    return torch.tensor(index_resampled.flatten()), torch.tensor(y_resampled.flatten())


def evaluate_img(model: torch.nn.Module, input_img: torch.Tensor,
                 device: torch.device = torch.device("cpu")) -> int:
    """
    Evaluates a single image using the trained model.

    :param model: Trained PyTorch model.
    :param input_img: Input image tensor (single image).
    :param device: Computation device (CPU/GPU). Default is CPU.
    :return: Predicted class label as an integer.
    """
    model.eval()  # Set model to evaluation mode

    # Ensure input is in batch format
    if input_img.dim() == 3:
        input_img = input_img.unsqueeze(0)

    input_img = input_img.to(device)  # Move to correct device
    model.to(device)  # Ensure model is on the correct device

    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_img)
        _, pred = torch.max(output, 1)  # Get the class with the highest probability

    return pred.item()  # Convert to Python int
