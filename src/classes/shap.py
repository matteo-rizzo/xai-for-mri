import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Subset

import shap
from src.classes.dataset import MRISubset, MRIDataset
from src.classes.models import ResNet50variant


class ImageProcessor:
    """Handles image format conversions between NHWC (TensorFlow) and NCHW (PyTorch)."""

    @staticmethod
    def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
        """Converts NCHW (PyTorch format) to NHWC (TensorFlow format)."""
        if x.dim() == 4:
            return x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            return x if x.shape[2] == 3 else x.permute(1, 2, 0)
        return x

    @staticmethod
    def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
        """Converts NHWC (TensorFlow format) to NCHW (PyTorch format)."""
        if x.dim() == 4:
            return x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            return x if x.shape[0] == 3 else x.permute(2, 0, 1)
        return x


class ShapExplainer:
    """Handles SHAP explanation generation and visualization."""

    def __init__(self, model_path: str, dataset: MRIDataset, test_indices: list,
                 output_path: str, device: torch.device = None):
        """
        Initializes the SHAP Explainer.

        :param model_path: Path to the pre-trained model.
        :param dataset: Full dataset.
        :param test_indices: Indices of test samples.
        :param output_path: Directory to save SHAP results.
        :param device: Computation device (CPU/GPU).
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.dataset = dataset
        self.test_dataset = MRISubset(Subset(dataset, test_indices), train_bool=False, transform=self._get_transforms())
        self.output_path = output_path
        self.class_names = ['healthy', 'affected']
        self.explainer = self._init_shap_explainer()
        self.img_names = [dataset[k][0].split('/')[-1] for k in test_indices]

        self._prepare_output_directory()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Loads a pre-trained ResNet50 variant."""
        model = ResNet50variant().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    @staticmethod
    def _get_transforms():
        """Returns the transformation pipeline for test images."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToDtype(torch.float32),
            transforms.ToTensor()
        ])

    def _prepare_output_directory(self):
        """Creates/clears the output directory for storing SHAP results."""
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

    def _init_shap_explainer(self) -> shap.Explainer:
        """Initializes the SHAP explainer using a blurred masker."""
        # Convert test dataset to NHWC format
        shap_test = torch.stack([img for img, _ in self.test_dataset])
        shap_test = ImageProcessor.nchw_to_nhwc(shap_test)

        # Define masker
        masker_blur = shap.maskers.Image("blur(64,64)", shap_test[0].shape)

        # Initialize SHAP explainer
        return shap.Explainer(self._predict, masker_blur, output_names=self.class_names, seed=11)

    def _predict(self, img: np.ndarray) -> torch.Tensor:
        """Performs a forward pass through the model and returns predictions."""
        self.model.eval()
        img = ImageProcessor.nhwc_to_nchw(torch.Tensor(img)).to(self.device)
        return self.model(img)

    def generate_shap_explanations(self, n_evals=5000, batch_size=50, topk=1):
        """
        Generates SHAP explanations for all test samples and saves them.

        :param n_evals: Number of evaluations for SHAP.
        :param batch_size: Batch size for SHAP evaluation.
        :param topk: Number of top predictions to consider.
        """
        numpy_output_path = os.path.join(self.output_path, 'shap_values')
        os.makedirs(numpy_output_path, exist_ok=True)

        shap_data = []
        for i in range(len(self.test_dataset)):
            input_img, _ = self.test_dataset[i]
            input_img = input_img.to(self.device)

            # Predict class
            predicted_class = torch.argmax(self.model(input_img.unsqueeze(0)).detach(), dim=1).item()

            # Compute SHAP values
            shap_values = self.explainer(
                input_img.unsqueeze(0),
                max_evals=n_evals,
                batch_size=batch_size,
                outputs=shap.Explanation.argsort.flip[:topk]
            )

            # Store SHAP values
            shap_values.data = shap_values.data.cpu().numpy()
            shap_data.append(shap_values)

            # Save SHAP image
            self._save_shap_image(shap_values, predicted_class, self.img_names[i])

            # Save SHAP numpy array
            np.save(os.path.join(numpy_output_path, f'{self.img_names[i]}.npy'), shap_values.values)

        print("SHAP explanations generated successfully.")

    def _save_shap_image(self, shap_values, predicted_class: int, img_name: str):
        """Saves SHAP image plots."""
        subdir = os.path.join(self.output_path, self.class_names[predicted_class])
        os.makedirs(subdir, exist_ok=True)

        # Plot and save SHAP image
        shap.image_plot(shap_values, show=False)
        plt.savefig(os.path.join(subdir, img_name))
        plt.close()
