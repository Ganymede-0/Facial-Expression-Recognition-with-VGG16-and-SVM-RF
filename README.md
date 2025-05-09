# Facial-Expression-Recognition-with-VGG16-and-SVM-RF
Overview:

This project investigates a hybrid machine learning approach to facial expression recognition using the FER-2013 dataset. By combining a pre-trained Convolutional Neural Network (VGG16) for feature extraction with traditional classifiers (Support Vector Machine and Random Forest), we aim to build an efficient and practical emotion classification model—especially suited for limited-resource environments like Google Colab.



Objectives:

- Extract deep visual features from facial images using VGG16 (without top classification layer).
- Apply PCA to reduce dimensionality and accelerate training.
- Compare classification performance of SVM and Random Forest.
- Handle dataset imbalance via oversampling.
- Evaluate both training and testing performance using accuracy, F1-score, and confusion matrix.



📂 Dataset
FER-2013 (Facial Expression Recognition 2013 from Kaggle)

- 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
- Grayscale 48×48 pixel images
- Total images: ~35,000 split into train/test
- Challenge: class imbalance, low resolution



Model Architecture
| Component         | Details                           |
| ----------------- | --------------------------------- |
| Feature Extractor | VGG16 (ImageNet weights)          |
| Preprocessing     | Resize to 224×224, convert to RGB |
| Feature Shape     | 25,088 (flattened from VGG16)     |
| Feature Reduction | PCA (n\_components = 500)         |
| Classifier #1     | Support Vector Machine (Linear)   |
| Classifier #2     | Random Forest (n=100)             |



Key Results
| Metric         | SVM (with PCA) | Random Forest (raw) |
| -------------- | -------------- | ------------------- |
| Test Accuracy  | \~54%          | \~53%               |
| Train Accuracy | \~62%          | \~99% (overfit)     |
| Best F1-Score  | \~53%          | \~52%               |

- SVM generalized better, especially after PCA
- RF overfitted due to high feature dimensionality
- Balanced training set improved results significantly

Conclusion:


This study demonstrates that combining pre-trained CNNs with classic classifiers offers a practical, resource-efficient solution for emotion recognition tasks—especially on small or imbalanced datasets like FER-2013. The best configuration was VGG16 + PCA + SVM, achieving competitive performance without the complexity of training deep networks from scratch.


📌 Future Work
- Fine-tuning the VGG16 model on FER-2013
- Applying feature selection methods beyond PCA
- Testing with other classifiers (e.g., XGBoost, LightGBM)
- Deploying a real-time inference system using OpenCV



