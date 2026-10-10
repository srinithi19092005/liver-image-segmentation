\# Liver and Tumor Segmentation Using U-Net



\## 1. Project Overview



This project implements automatic liver and tumor segmentation from CT images using a U-Net deep learning model.



The system is trained using the LiTS (Liver Tumor Segmentation) dataset and provides a Streamlit web application for uploading CT images and viewing predicted liver and tumor regions.



\---



\## 2. Objectives



The main objectives of this project are:



1\. Prepare and preprocess liver CT image data.

2\. Develop a U-Net based segmentation model.

3\. Segment the liver from CT images.

4\. Segment tumor regions from CT images.

5\. Evaluate the trained model using segmentation metrics.

6\. Develop a simple Streamlit application for model demonstration.



\---



\## 3. Dataset



The project uses the LiTS dataset.



The dataset contains:



\- CT image slices

\- Liver segmentation masks

\- Tumor segmentation masks



The images were converted/organized as PNG files for use with the PyTorch training pipeline.



Dataset structure:



```text

dataset/

└── LiTS/

\&#x20;   ├── train/

\&#x20;   │   ├── images/

\&#x20;   │   ├── liver\\\_masks/

\&#x20;   │   └── tumor\\\_masks/

\&#x20;   │

\&#x20;   ├── splits/

\&#x20;   │   ├── train.csv

\&#x20;   │   └── validation.csv

\&#x20;   │

\&#x20;   └── lits\\\_train.csv


