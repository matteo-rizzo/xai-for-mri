import os
import shutil

import cv2
import numpy as np
import torch

from src.config import ID_TO_NAME


class ExplainabilityEnsemble:
    def __init__(self, models, input_path, output_path, test_index, data, device='cpu', size=7):
        self.input_path = input_path
        self.output_path = output_path
        self.test_index = test_index
        self.data = data
        self.device = device
        self.size = size

        self.models = models
        self.img_names, self.labels = self.get_image_info()

        # Clear and create the output directory
        self.prepare_output_dir()

    def get_image_info(self):
        """Extract image names and labels."""
        img_names = []
        labels = []
        for k in self.test_index:
            name, label = self.data[k]
            labels.append(label)
            img_names.append(name.split('/')[-1])  # Extract filename from path
        return img_names, labels

    def prepare_output_dir(self):
        """Prepare the output directory structure."""
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)

    def process_image(self, model, img_name, n):
        """Process the image using a specific model."""
        img_orig, _ = self.data[self.test_index[n]]
        rgb_img = img_orig.repeat(3, 1, 1).numpy().transpose((1, 2, 0))
        pred = self.evaluate_img(model, img_orig.unsqueeze(0)).item()

        # Load explanation maps for the current model
        gradcam = np.load(os.path.join(self.input_path, "cam_array", model, "GradCAM", img_name + '.npy'))
        gradcampp = np.load(os.path.join(self.input_path, "cam_array", model, "GradCAMPlusPlus", img_name + '.npy'))
        hirescam = np.load(os.path.join(self.input_path, "cam_array", model, "HiResCAM", img_name + '.npy'))

        shap5000 = np.squeeze(
            np.load(os.path.join(self.input_path, "Shap", f"{model}-5000", "shap_values", img_name + '.npy'))[:, :, :,
            :, 0][
                0])

        occlusion_file = os.path.join(self.input_path, "Occlusion_res", model,
                                      f"{img_name}.npy" if pred == 1 else f"{img_name}-2.npy")
        occlusion = np.load(occlusion_file)

        # Collect explainability components for ensemble
        explainability = [
            self.discretize(gradcam),
            self.discretize(gradcampp),
            self.discretize(hirescam),
            self.discretize(shap5000)
        ]
        if np.any(occlusion >= 1):
            explainability.append(self.discretize(occlusion))

        return rgb_img, explainability

    def discretize(self, explanation_map):
        """Discretize the explainability map."""
        return np.array([explanation_map[i:i + self.size, j:j + self.size]
                         for i in range(0, explanation_map.shape[0], self.size)
                         for j in range(0, explanation_map.shape[1], self.size)])

    def evaluate_img(self, model, img):
        """Evaluate the image with the given model."""
        model.eval()
        with torch.no_grad():
            output = model(img.to(self.device))
        return output

    @staticmethod
    def generate_final_image(rgb_img, explainability):
        """Generate a final image by blending explainability heatmap."""
        # Apply ensemble algorithm to combine explainability maps
        res = ensemble_algorithm(explainability)

        # Generate heatmap from ensemble results
        heatmap = cv2.applyColorMap(np.uint8(255 * res), cv2.COLORMAP_PARULA)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        # Blend heatmap with original image
        final_img = cv2.addWeighted(rgb_img, 0.5, heatmap, 0.5, 0)
        return final_img

    def save_results(self, img_name, final_img, predicted_class_index):
        """Save the final blended image."""
        subdir = ID_TO_NAME[self.labels[self.test_index.index(img_name)]]
        final_output_path = os.path.join(self.output_path, predicted_class_index, subdir)

        os.makedirs(final_output_path, exist_ok=True)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{final_output_path}/{img_name}', final_img)

    def run(self):
        """Run the ensemble process for all test images."""
        for n, img_name in enumerate(self.img_names):
            for model_name, model in self.models:
                rgb_img, explainability = self.process_image(model_name, img_name, n)

                # Apply ensemble method to combine explainability maps
                final_img = self.generate_final_image(rgb_img, explainability)

                # Save the result
                self.save_results(img_name, final_img, model_name)
