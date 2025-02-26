import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM


class OcclusionMapGenerator:
    """
    Class to perform hierarchical occlusion and generate occlusion maps for image regions
    to evaluate the importance of each region for a target class.
    """

    def __init__(self, model, device, target_layers=None):
        self.model = model.to(device)
        self.device = device
        self.target_layers = target_layers if target_layers else [model.conv]
        self.cam_methods = {
            "GradCAM": GradCAM,
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "HiResCAM": HiResCAM
        }

    def hierarchical_occlusion(self, image, target_class, stride, window_size, min_window_size, x_start, y_start, x_end,
                               y_end, zero=False):
        """
        Applies hierarchical occlusion to evaluate the importance of each region of the image for the target class.
        """
        output, areas = self.class_occlusion(image, target_class, stride, window_size, x_start, y_start, x_end, y_end,
                                             None, zero)

        while len(areas) > 0 and window_size > min_window_size:
            stride = max(stride // 2, 1)
            window_size = window_size // 2
            diff_map, areas = self.class_occlusion(image, target_class, stride, window_size, x_start, y_start, x_end,
                                                   y_end, areas, zero)
            output += diff_map

        return output

    def class_occlusion(self, image, target_class, stride, window_size, x_start, y_start, x_end, y_end, old_areas,
                        zero):
        """
        Performs class occlusion and computes the difference map by occluding regions.
        """
        original_label = self.evaluate_img(image.unsqueeze(0))

        _, H, W = image.shape
        diff_map = np.zeros((H, W), dtype=float)
        new_areas = []

        for i in range(y_start, y_end, max(stride, 1)):
            y_occlusion_end = min(y_end, i + window_size)
            for j in range(x_start, x_end, max(stride, 1)):
                x_occlusion_end = min(x_end, j + window_size)

                if self.is_inside(i, j, old_areas, window_size * 2):
                    occluded_image = image.clone()

                    if zero:
                        occluded_image[:, i:y_occlusion_end, j:x_occlusion_end] = 0
                    else:
                        occluded_image[:, i:y_occlusion_end, j:x_occlusion_end] = 1 if target_class != 1 else 0

                    occluded_label = self.evaluate_img(occluded_image.unsqueeze(0))

                    diff = abs(original_label - occluded_label)
                    diff_map[i:y_occlusion_end, j:x_occlusion_end] = np.maximum(diff, diff_map[i:y_occlusion_end,
                                                                                      j:x_occlusion_end])

                    if diff == 1:
                        new_areas.append((i, j))

        return diff_map, new_areas

    @staticmethod
    def is_inside(y, x, areas, window_size):
        """
        Checks if the given position (y, x) is inside any of the previously occluded areas.
        """
        if areas is None:
            return True
        for (ay, ax) in areas:
            if ay <= y < ay + window_size and ax <= x < ax + window_size:
                return True
        return False

    def evaluate_img(self, image):
        """
        Evaluate the model on an image and return the predicted class.
        """
        output = self.model(image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()


class OcclusionHeatmap:
    """
    Class to manage the application of occlusion heatmaps and visualization.
    """

    def __init__(self, model, test_data, data, model_variant="Resnet18v", path_to_models="models"):
        self.model = model
        self.test_data = test_data
        self.data = data
        self.model_variant = model_variant
        self.path_to_models = path_to_models
        self.stats = {"p1": 0, "p0": 0, "count_affected": 0, "count_healthy": 0}

    def apply_occlusion_heatmap(self):
        """
        Applies the occlusion heatmap to the test dataset images using the specified model.
        """
        for i in range(len(self.test_data)):
            img, label = self.test_data[i]
            test_idx = self.test_data.subset.indices[i]
            path, _ = self.data[test_idx]
            name = path.split('\\')[-1]

            _, H, W = img.shape
            pred = self.evaluate_img(img.unsqueeze(0)).item()

            rgb_img = img.repeat(3, 1, 1).numpy().transpose((1, 2, 0))

            occlusion_map = self.load_occlusion_map(name, pred, H, W)
            if np.any(occlusion_map >= 1):
                self.stats["count_affected" if pred == 1 else "count_healthy"] += 1
            else:
                occlusion_map = np.zeros((H, W), dtype=float)

            self.stats["p1" if pred == 1 else "p0"] += 1
            self.display_heatmap(rgb_img, occlusion_map, name)

        print(
            f"Pred 1: {self.stats['p1']}, Pred 0: {self.stats['p0']}, Count 1: {self.stats['count_affected']}, Count 0: {self.stats['count_healthy']}")

    def evaluate_img(self, image):
        """
        Evaluates the model on an image and returns the predicted class.
        """
        output = self.model(image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

    def load_occlusion_map(self, name, pred, H, W):
        """
        Load the occlusion map for the given image name and prediction.
        """
        suffix = "" if pred == 1 else "-2"
        map_path = os.path.join(self.path_to_models, self.model_variant, f"{name}{suffix}.npy")
        if os.path.exists(map_path):
            return self.normalize(np.load(map_path))
        else:
            return np.zeros((H, W), dtype=float)

    def display_heatmap(self, rgb_img, occlusion_map, name):
        """
        Displays the heatmap for the occluded image.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * occlusion_map), cv2.COLORMAP_PARULA)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        final_img = cv2.addWeighted(rgb_img, 1, heatmap, 0.5, 0)
        final_img = self.normalize(final_img)
        plt.imshow(final_img)
        plt.title(name)
        plt.colorbar()
        plt.show()

    @staticmethod
    def normalize(image):
        """
        Normalizes the image for display.
        """
        return image / np.max(image) if np.max(image) > 0 else image
