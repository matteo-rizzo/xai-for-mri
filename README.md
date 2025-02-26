# Assessing the Value of XAI for MRI

## Overview

This repository contains code for training, evaluating, and explaining an MRI image classification model using ResNet.
The model is trained to classify MRI images as either "healthy" or "affected" using a variety of machine learning and
explainability techniques, including GradCAM, SHAP, and occlusion-based methods.

## Features

- **Cross-validation**: Uses stratified group k-fold validation for robust model evaluation.
- **ResNet50-based classification**: Implements ResNet50 as the primary model.
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
├── cam-vs-occlusion.ipynb      # Notebook for comparing occlusion and GradCAM results
├── ensemble.ipynb              # Notebook for ensemble explainability methods
├── requirements.txt            # Required dependencies
└── src
    ├── classes
    │   ├── Dataset.py          # Dataset processing and augmentation
    │   └── Models.py           # ResNet50 and ResNet18 model definitions
    ├── config.py               # Configuration file for training settings
    └── functions
        ├── train_eval.py       # Training and evaluation pipeline
        └── utils_train.py      # Utility functions (normalization, visualization, etc.)
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

To generate occlusion-based maps:

```sh
jupyter notebook cam-vs-occlusion.ipynb
```

#### Generate Explainability Ensemble

To generate SHAP and CAM ensemble:

```sh
jupyter notebook ensemble.ipynb
```

## Model Training Details

- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Cross-validation**: 10-fold stratified group k-fold
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## Explainability Techniques

- **SHAP Values**: Feature importance analysis using SHAP.
- **GradCAM & GradCAM++**: Visualizing CNN attention in MRI scans.
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

