# Stroke Prediction (PyTorch DNN)

A simple PyTorch project for predicting stroke risk using a deep neural network on an imbalanced healthcare dataset.

Built in a single Jupyter Notebook (`Stroke Predict Pytorch DNN.ipynb`) with weighted sampling, early stopping (on PR-AUC), and evaluation metrics. This serves mainly as a comparison to the XGBoost version.

## What it does
- Loads and preprocesses the stroke prediction dataset (healthcare-dataset-stroke-data.csv)
- Handles class imbalance with WeightedRandomSampler
- Trains a multi-layer DNN with BatchNorm and Dropout
- Uses BCE loss, AdamW, CosineAnnealingWarmRestarts scheduler, and early stopping on validation PR-AUC
- Evaluates on test set with PR-AUC, Precision, Recall, F1

## Tech Stack
- PyTorch + TorchMetrics
- Scikit-learn (for splits and scaling)
- Google Colab (GPU-ready)

## Quick Note
This DNN implementation is for benchmarking against XGBoost on imbalanced binary classification.

## Result
Early stopping triggered after 28 epochs (best val PR-AUC: 0.3663).
Test set (final):

- PR-AUC: 0.2358
- Precision: 0.1098
- Recall: 0.7200
- F1: 0.1905

---

Made for learning purposes.
