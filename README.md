# Facial Landmark Detection using Computer Vision

This project implements a facial landmark detection system using computer vision and machine learning to locate and predict five facial keypoints (eyes, nose, and mouth corners) in images.  
Developed as part of the *Applied Machine Learning* module at the University of Sussex (2025), it demonstrates how feature extraction, data augmentation, and regression models can be combined to perform robust face alignment.

---

## Overview

Facial landmark detection is a key task in computer vision, forming the foundation for applications such as face recognition, emotion analysis, and augmented reality.  
This project applies Histogram of Oriented Gradients (HOG) features and a Ridge Regression (L2 regularised) model to predict facial landmarks from grayscale images.

Through systematic experimentation, this approach achieved a Mean Point Error (MPE) of 4.36 pixels on the validation set (measured on 256×256 images), showing strong localisation accuracy and generalisation.

---

## Implementation Details

Language: Python  
Libraries: OpenCV, NumPy, pandas, scikit-learn, matplotlib, scikit-image  

Key steps:
1. Image preprocessing (grayscale conversion, resizing to 96×96 pixels, and normalisation).  
2. Data augmentation (horizontal flipping with landmark swapping and brightness variation).  
3. Feature extraction using Histogram of Oriented Gradients (HOG).  
4. Training a Ridge Regression model with cross-validated α = 75.0.  
5. Scaling predicted coordinates back to the original image dimensions (256×256).  

Dataset used:  
- `face_alignment_training_images.npz` (images + labels)  
- `face_alignment_test_images.npz` (images only)  

---

## Model Performance

| Model | Features | Augmentation | Mean Point Error (pixels) |
|:------|:----------|:-------------|:--------------------------:|
| SIFT + Ridge Regression | SIFT | None | 8.77 |
| HOG + Ridge Regression | HOG | None | 5.12 |
| HOG + Ridge Regression | HOG | Flip + Brightness | **4.36** |

The final model achieved a mean point error (MPE) of 4.36 px, with a median error of 3.23 px and 95% of predictions within 10.49 px of ground truth.  
These results demonstrate the effectiveness of HOG features and augmentation in capturing facial structure under varied lighting and pose conditions.

---

## Visual Results

<p align="center">
  <img src="./images/Facial%20Landmark%20picture%20.png" width="500"/>
</p>
<p align="center"><em>Predicted (red) vs Ground Truth (green) facial landmarks on a sample image.</em></p>

<p align="center">
  <img src="./images/Facial%20landmark%20boxplot%20image.png" width="500"/>
</p>
<p align="center"><em>Boxplot of landmark prediction error distribution across validation data.</em></p>

---

## How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Ruairibradley/Facial-Landmark-Detection.git
   cd Facial-Landmark-Detection



2. Open the project notebook:
Launch Jupyter Notebook or another Python IDE (VS Code, PyCharm, or Google Colab).
Open the file:

AML_Code_Task1.ipynb


Follow the notebook workflow:
Run each cell sequentially to:
Load and preprocess the dataset
Extract HOG features
Train and validate the Ridge Regression model
Visualise predicted vs actual facial landmarks
View the results:
Validation visualisations will display predicted landmarks (red) vs ground truth (green).
Error metrics and plots (histograms, boxplots) will be generated automatically.
Experiment:
You can adjust HOG parameters, α regularisation, or augmentation probability within the notebook to explore different configurations.

## Full Report: 

A detailed technical report covering the dataset, methodology, experiments, and performance analysis is available here:

[Download Full Report (PDF)](./Report.pdf)

The report includes:
Description of preprocessing and augmentation pipeline
Details on HOG feature extraction and Ridge Regression tuning
Quantitative results with validation metrics and error analysis
Discussion of strengths, limitations, and potential future improvements
