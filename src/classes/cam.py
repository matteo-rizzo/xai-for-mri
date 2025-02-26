import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.functions.utils_train import evaluate_img


class CAMGenerator:
    """
    Class for generating and saving Class Activation Maps (CAMs) for a given dataset.
    """

    def __init__(self, model: torch.nn.Module, cam: GradCAM, output_path: str, class_mapping: dict):
        """
        Initializes the CAMGenerator.

        :param model: Trained PyTorch model.
        :param cam: GradCAM object.
        :param output_path: Directory to save CAM images or numpy arrays.
        :param class_mapping: Dictionary mapping class indices to names.
        """
        self.model = model
        self.cam = cam
        self.output_path = output_path
        self.class_mapping = class_mapping

        os.makedirs(self.output_path, exist_ok=True)

    def generate_cam_image(
            self, img: torch.Tensor, target: Optional[int] = None, transform=None, plot: bool = False
    ) -> np.ndarray:
        """
        Generates a Class Activation Map (CAM) for a given image.

        :param img: Input image tensor.
        :param target: Target class label. If None, uses model prediction.
        :param transform: Optional image transformation function.
        :param plot: If True, displays the CAM visualization.
        :return: Processed CAM image.
        """
        img = img.unsqueeze(0)  # Add batch dimension
        target = target or evaluate_img(self.model, img)
        grayscale_cam = self.cam(input_tensor=img, targets=[ClassifierOutputTarget(target)])[0]

        img = transform(img.squeeze()) if transform else img.squeeze()
        rgb_img = img.repeat(3, 1, 1).numpy().transpose((1, 2, 0))
        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        if plot:
            self._plot_cam(rgb_img, cam_img)

        return cam_img

    @staticmethod
    def _plot_cam(rgb_img: np.ndarray, cam_img: np.ndarray):
        """
        Plots the original and CAM-processed images side by side.

        :param rgb_img: Original RGB image.
        :param cam_img: Processed CAM image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(rgb_img)
        axes[0].axis("off")
        axes[1].imshow(cam_img)
        axes[1].axis("off")
        plt.show()

    def save_cam_images(self, test_dataset, data_mapping, as_numpy: bool = False):
        """
        Generates and saves CAM images or numpy arrays for a dataset.

        :param test_dataset: Dataset subset.
        :param data_mapping: Dictionary mapping indices to image paths and labels.
        :param as_numpy: If True, saves CAM as numpy arrays instead of images.
        """
        for i, (img, _) in enumerate(test_dataset):
            test_idx = test_dataset.subset.indices[i]
            path, _ = data_mapping[test_idx]
            filename = os.path.basename(path)
            pred = evaluate_img(self.model, img.unsqueeze(0))

            if as_numpy:
                grayscale_cam = self.cam(input_tensor=img.unsqueeze(0), targets=[ClassifierOutputTarget(pred)])[0]
                np.save(os.path.join(self.output_path, f"{filename}.npy"), grayscale_cam)
            else:
                cam_img = self.generate_cam_image(img, target=pred)
                self._save_image(cam_img, pred, filename)

        print("CAM generation and saving completed.")

    def _save_image(self, cam_img: np.ndarray, pred: int, filename: str):
        """
        Saves the generated CAM image to the appropriate directory.

        :param cam_img: Processed CAM image.
        :param pred: Predicted class index.
        :param filename: Image filename.
        """
        subdir = os.path.join(self.output_path, self.class_mapping[pred])
        os.makedirs(subdir, exist_ok=True)
        cv2.imwrite(os.path.join(subdir, filename), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
