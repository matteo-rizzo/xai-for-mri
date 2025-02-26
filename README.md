# Assessing the Value of XAI for MRI

## Overview

This repository contains code for training, evaluating, and explaining an MRI image classification model using ResNet.
The model is trained to classify MRI images as either "healthy" or "affected" using a variety of machine learning and
explainability techniques, including GradCAM, SHAP, and occlusion-based methods.

## Features

- **Cross-validation**: Uses stratified group k-fold validation for robust model evaluation.
- **ResNet-based classification**: Implements ResNet18 and ResNet50 as classification models.
- **Data Augmentation**: Includes Gaussian blur and color jitter transformations.
- **Ensemble Methods**: Combines multiple explainability methods for model interpretation.
- **SHAP and GradCAM explainability**: Uses SHAP values and GradCAM to visualize model decisions.
- **Hierarchical occlusion analysis**: Identifies important image regions by systematically masking parts of the image.
- **Evaluation Metrics**: Computes accuracy, precision, recall, and F1-score.
- **Automatic Result Saving**: Stores model evaluation metrics in a JSON file.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── notebooks
│   ├── cam-methods.ipynb      # Notebook for GradCAM, GradCAM++, and HiResCAM
│   ├── ensemble.ipynb         # Notebook for ensemble explainability methods
│   ├── occlusion.ipynb        # Notebook for occlusion-based explainability
│   ├── shap.ipynb             # Notebook for SHAP-based explainability
│   └── training.ipynb         # Notebook for model training and evaluation
├── requirements.txt           # Required dependencies
└── src
    ├── classes
    │   ├── cam.py             # CAM methods (GradCAM, GradCAM++, HiResCAM)
    │   ├── dataset.py         # Dataset processing and augmentation
    │   ├── ensemble.py        # Ensemble explainability algorithm
    │   ├── models.py          # ResNet50 and ResNet18 model definitions
    │   ├── occlusion.py       # Occlusion-based explainability
    │   ├── shap.py            # SHAP explainability methods
    │   └── training.py        # Training pipeline
    ├── config.py              # Configuration file for training settings
    └── functions
        ├── utils_ensemble.py  # Utility functions for ensemble methods
        └── utils_train.py     # Utility functions (normalization, visualization, etc.)
```

## Installation

### Prerequisites

Ensure you have Python 3.8 or later installed.

### Install dependencies

```sh
pip install -r requirements.txt
```

## Usage

### Running the Notebooks

All training, evaluation, and explainability processes are handled through the Jupyter notebooks. Open and execute the
following notebooks based on your needs:

#### Train and Evaluate the Model

To train and evaluate the model:

```sh
jupyter notebook notebooks/training.ipynb
```

#### Generate SHAP Explanations

To generate SHAP-based explainability results:

```sh
jupyter notebook notebooks/shap.ipynb
```

#### Generate CAM Explanations

To generate GradCAM, GradCAM++, and HiResCAM results:

```sh
jupyter notebook notebooks/cam-methods.ipynb
```

#### Generate Occlusion-based Explanations

To generate occlusion-based maps:

```sh
jupyter notebook notebooks/occlusion.ipynb
```

#### Generate Explainability Ensemble

To generate SHAP and CAM ensemble:

```sh
jupyter notebook notebooks/ensemble.ipynb
```

## Model Training Details

- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Cross-validation**: 10-fold stratified group k-fold
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## Explainability Techniques

- **SHAP Values**: Feature importance analysis using SHAP.
- **GradCAM, GradCAM++, HiResCAM**: Visualizing CNN attention in MRI scans.
- **Hierarchical Occlusion Analysis**: Identifying key regions in the MRI scans by occluding parts of the image.

## Results

After training, results are stored in:

```
src/models/final-resnet50v/results.json
```

Example output:

```json
{
  "title": "final-resnet50v",
  "learning rate": 0.001,
  "momentum": 0.9,
  "epochs": 50,
  "final": {
    "accuracy": 0.92,
    "precision": 0.90,
    "recall": 0.88,
    "f1": 0.89
  }
}
```

## License

This project is licensed under the MIT License.