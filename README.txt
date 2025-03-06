# ECG Analysis and Prediction Pipeline

## Overview

This project implements a dual machine learning pipeline for analyzing ECG (Electrocardiogram) data and predicting COVAS values. The system uses a two-step approach:

1. A binary classifier that predicts whether a COVAS value will be zero or non-zero
2. A regressor that predicts the actual COVAS value for non-zero cases

This approach optimizes both classification accuracy and regression performance.

## Features

- ECG signal processing and feature extraction
- Heart Rate Variability (HRV) analysis
- Frequency domain analysis of ECG signals
- Feature engineering with signal derivatives
- Dual prediction model (classifier + regressor)
- Comprehensive evaluation metrics and visualizations
- Model interpretability with SHAP values
- C code export capability for embedded systems

## Requirements

- Python 3.6+
- Required packages:
  - matplotlib
  - seaborn
  - pandas
  - numpy
  - scikit-learn
  - scipy
  - shap

