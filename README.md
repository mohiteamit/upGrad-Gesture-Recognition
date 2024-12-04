# upGrad-Gesture-Recognition

## Overview

The project is a comprehensive exploration of various machine learning models for recognizing 5 hand gestures. These gestures are extracted from sequences of frames captured during hand movement. The project evaluates both non-pretrained and pretrained models to identify the most effective and efficient architecture for gesture recognition.

This repository combines modern machine learning techniques with feature extraction methods, such as **Conv3D** and **GRU**, along with pretrained models like **MobileNetV2** and **MediaPipe Hand Keypoints**.

---

## Task and Methodology

The goal is to classify gestures based on their sequences. We employ spatial-temporal feature extraction techniques using:

- **Non-pretrained models** such as Conv3D, Conv2D with GRU, and Conv2D with LSTM.
- **Pretrained models** using MobileNetV2, MobileNetV3Small, and MediaPipe for feature extraction.

Each model undergoes rigorous evaluation based on accuracy, loss stability, and generalization capabilities across training and validation datasets.

---

## Summary of Results

### Best Models Summary

| **Category**        | **Model**                     | **Key Hyperparameters**                                                                 | **Results**                             | **Why This Model Stands Out**                                                                                                                                                                                                                      |
|----------------------|--------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Non-Pretrained**   | Conv3D                        | 3 Conv3D layers (filters: 32, 64, 128), GlobalAvgPooling, Dense (128 units, dropout 50%) | Accuracy: Train 93.82%, Val 89.0%        | Conv3D achieves consistent performance with stable loss and accuracy trends. Its ability to learn spatial and temporal features simultaneously, without relying on pretrained weights, makes it a robust option. The stability suggests potential for improvement with additional training.           |
| **Pretrained**       | GRU with MediaPipe Keypoints  | GRU (64 units), Flatten, Dense (5 units), Dropout (50%), MediaPipe Hand Keypoints       | Accuracy: Train 95.5%, Val 99.0%         | Leveraging MediaPipe’s pretrained keypoint extractor reduces input complexity, allowing the lightweight GRU-based model to achieve exceptional accuracy and efficiency. Its simplicity and computational efficiency make it ideal for real-time applications while maintaining robust generalization. |

---

## MediaPipe Hand Keypoints

The **GRU with MediaPipe Keypoints** model utilizes **MediaPipe Hands**, a pretrained feature extractor, to identify 21 keypoints of the hand. These keypoints (x, y, z coordinates) serve as lightweight input features, replacing raw image sequences for temporal modeling.

### Example Visualization

The following image showcases how MediaPipe marks hand keypoints during gesture recognition:

![Mediapipe hand reduced image to 21 key points](image.png)
---

## Environment and Training

The models were trained and tested using the following environments:

## Repository Structure

- **`models/`**: Contains scripts defining various model architectures.
- **`data/`**: Placeholder for gesture datasets.
- **`utilities.py`**: Helper functions for data preprocessing and evaluation.
- **`results/`**: Performance graphs and metrics for all evaluated models.
- **`README.md`**: Detailed project overview, results summary, and usage instructions.

---

## To clone and use repository

```bash
   git clone https://github.com/your-repo/upGrad-Gesture-Recognition.git
   cd upGrad-Gesture-Recognition
```

## Requirements
```bash
pip install -r requirements.txt
```

```plaintext
tensorflow
mediapipe
opencv-python
numpy
pandas
matplotlib
scikit-learn
seaborn
tensorboard
scipy
Pillow
gdown
```