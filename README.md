# 🌍 Conformal Prediction for Remote Sensing (6 Bands)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-orange?style=for-the-badge&logo=tensorflow)
![Uncertainty Quantification](https://img.shields.io/badge/UQ-Split%20CP%20%7C%20Class--wise%20CP%20%7C%20Cluster--wise%20CP-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

A cutting-edge machine learning repository dedicated to **Pixel-Wise Classification of Multispectral Remote Sensing Data**. This project integrates deep learning architectures (CNNs, Vision Transformers, Global Filter Networks) with advanced **Conformal Prediction (CP)** techniques to generate rigorous uncertainty maps and guaranteed prediction sets for satellite imagery.

---

## 📑 Table of Contents (Navigation)

1. [📌 Project Overview](#-project-overview)
2. [📂 Repository Structure](#-repository-structure)
3. [📊 Dataset Details](#-dataset-details)
4. [🛠️ Workflow & Methodology](#-workflow--methodology)
    - [Phase 1: Deep Learning Models](#phase-1-deep-learning-models)
    - [Phase 2: Split Conformal Prediction (SCP)](#phase-2-split-conformal-prediction-scp)
    - [Phase 3: Class-wise & Cluster-wise CP](#phase-3-class-wise--cluster-wise-cp)
    - [Phase 4: Uncertainty Visualization](#phase-4-uncertainty-visualization)

---

## 📌 Project Overview

Remote sensing data classification often lacks reliability measures. This repository addresses this by moving beyond simple "argmax" class predictions.
* **Objective**: Classify 6-band multispectral pixels into semantic categories (e.g., Vegetation, Water, Urban).
* **Key Innovation**: Application of three distinct Conformal Prediction algorithms to handle data imbalance and provide statistical guarantees (e.g., 95% confidence).
* **Outputs**: Prediction sets (multiple classes per pixel if uncertain) and heatmap visualizations of model uncertainty.

---

## 📂 Repository Structure

The repository is organized by model architecture and predictive head configuration.

### 1. 🗂️ [Data](./Data)
Contains the raw multispectral and ground truth data.
* **[`data.csv`](./Data/data.csv)**: 6-channel pixel values (Band1 - Band6).
* **[`ref.csv`](./Data/ref.csv)**: Ground Truth (GT) labels for semantic segmentation.

### 2. 🧠 Single Head Models
Standard deep learning architectures with a single classification output head.
* **[`CNN Model/`](./Single%20Head/CNN%20Model)**: Implements **AlexNet** adapted for 6-channel input.
* **[`Vision Transformers/`](./Single%20Head/Vision%20Transformers)**: Implements **ViT + U-Net** hybrids for capturing spatial and spectral dependencies.
* **[`Global Filter Networks/`](./Single%20Head/Global%20Filter%20Networks)**: Frequency-domain mixing models.

### 3. 🧩 Multi Head Models
Architectures designed with multiple heads, potentially for ensemble-based uncertainty or multi-task learning.
* **[`CNN Model/`](./Multi%20Head/CNN%20Model)**
* **[`Vision Transformers/`](./Multi%20Head/Vision%20Transformers)**
* **[`Global Filter Networks/`](./Multi%20Head/Global%20Filter%20Networks)**

Each model folder contains:
* **`*.ipynb`**: The main training and calibration notebook.
* **`conformal_reports.xlsx`**: detailed logs of coverage, set sizes, and run metrics.

---

## 📊 Dataset Details

The input data consists of multispectral imagery suitable for land-cover classification.

| Feature | Description | Range |
| :--- | :--- | :--- |
| **Band 1 - Band 6** | Reflectance values across 6 spectral bands | 0 - 255 (normalized in code) |
| **GT (Ground Truth)** | Integer class labels (0, 1, 2...) representing land cover types | Categorical |

**Preprocessing**:
* Data is reshaped into `(Height, Width, Bands)`.
* **Patch Extraction**: The code extracts `P_S x P_S` (e.g., 9x9) patches around each pixel to utilize spatial context during classification.

---

## 🛠️ Workflow & Methodology

### Phase 1: Deep Learning Models
We train three state-of-the-art backbones to extract features from multispectral patches:
1.  **AlexNet (CNN)**: Captures local spatial features.
2.  **ViT (Vision Transformer)**: Captures global context using self-attention mechanisms.
3.  **GFN (Global Filter Network)**: Uses Fast Fourier Transforms (FFT) for global mixing without heavy attention computation.

### Phase 2: Split Conformal Prediction (SCP)
Standard marginal coverage guarantee.
* **Calibration**: A hold-out calibration set is used to compute non-conformity scores (1 - softmax probability).
* **Quantile**: A score threshold $\hat{q}$ is derived such that 95% of true labels are included.
* **Outcome**: Prediction sets valid on *average* across the entire image.

### Phase 3: Class-wise & Cluster-wise CP
Remote sensing data is often highly imbalanced (e.g., vast water bodies vs. small buildings).
* **Class-wise CP (CwCP)**: Calibrates a separate threshold for *each class*. Ensures that rare classes (minorities) are not under-covered.
* **Cluster-wise CP (ClCP)**: Uses **K-Means clustering** on the model's embedding space to group spectrally similar classes together, then calibrates within these clusters. This balances granularity and statistical strength.

### Phase 4: Uncertainty Visualization
Instead of a single label map, we generate:
* **Set Size Maps**: A heatmap showing how many classes are in the prediction set for each pixel. (Size > 1 = Uncertain).
* **Voronoi Analysis**: Visualizing decision boundaries in the feature space.
* **Excel Reports**: Automated generation of `conformal_reports.xlsx` containing coverage plots and sample images.

---
