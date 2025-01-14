# Assignment 3: Image Classification with CNNs and Transfer Learning

## Overview

This repository contains the implementation of image classification using a custom **Convolutional Neural Network (CNN)** and **transfer learning** with pre-trained models such as **ResNet50** and **MobileNetV2**. The primary goal is to classify images from the Animal-10 dataset into 10 classes using deep learning techniques.

### Objectives

1. Design and implement a CNN from scratch for image classification.
2. Utilize transfer learning with ResNet18 and MobileNetV2 to improve performance.
3. Compare and analyze the advantages, disadvantages, and performance of models trained from scratch versus pre-trained models.
4. Evaluate models on metrics such as accuracy, precision, recall, and F1-score.

---

## Highlights

- **Dataset:** The Animal-10 dataset, containing images from 10 animal classes.
- **Preprocessing:** Applied transformations like resizing, normalization, random rotations, and horizontal flips to augment the data and improve generalization.
- **Custom CNN Architecture:** Designed a CNN with 4 convolutional blocks followed by adaptive pooling and fully connected layers.
- **Transfer Learning:** Fine-tuned pre-trained ResNet18 and MobileNetV2 models for the classification task.
- **Evaluation:** Models evaluated on test data with metrics like accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization:** Training and validation loss trends, confusion matrices, and performance comparison across models.

---

## Results Summary

### Model Performance Comparison

| Model             | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| CNN (from scratch) | 67%      | 69%       | 67%    | 67%      |
| ResNet18           | 74%      | 79%       | 74%    | 74%      |
| MobileNetV2        | 83%      | 84%       | 83%    | 83%      |

### Key Insights

1. **CNN from Scratch:**
   - Moderate performance but required significantly more training time and resources.
   - Struggled to generalize for some classes due to limited dataset size and model capacity.

2. **ResNet50 (Transfer Learning):**
   - Balanced performance with good accuracy and computational efficiency.
   - Fine-tuning specific layers helped adapt pre-trained features to the dataset.

3. **MobileNetV2 (Transfer Learning):**
   - Achieved the best performance with high accuracy and efficiency.
   - Lightweight model suitable for resource-constrained environments.

---

## Repository Structure

```plaintext
.
├── assignment3_template.ipynb  # Jupyter notebook with the full implementation
├── pa3_subset_animal/          # Dataset directory (Animal-10 dataset)
├── best_cnn_model.pth          # Trained CNN model weights
├── best_resnet18_model.pth     # Trained ResNet18 model weights
├── best_mobilenet_model.pth    # Trained MobileNetV2 model weights
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
