# Liver and Tumor Segmentation Using U-Net

## 1. Project Overview

This project implements automatic liver and tumor segmentation from CT images using a U-Net deep learning model.

The system uses the LiTS (Liver Tumor Segmentation) dataset and provides a Streamlit web application for uploading CT images and viewing predicted liver and tumor regions.

## 2. Objectives

The main objectives of this project are:

1. Prepare and preprocess liver CT image data.
2. Develop a U-Net-based segmentation model.
3. Segment the liver from CT images.
4. Segment tumor regions from CT images.
5. Evaluate the trained model using segmentation metrics.
6. Develop a simple Streamlit application for model demonstration.

## 3. Dataset

This project uses the LiTS (Liver Tumor Segmentation) dataset.

The dataset contains:

- CT image slices
- Liver segmentation masks
- Tumor segmentation masks

The images were converted and organized as PNG files for use with the PyTorch training pipeline.

### Dataset Structure

```text
dataset/
└── LiTS/
    ├── train/
    │   ├── images/
    │   ├── liver_masks/
    │   └── tumor_masks/
    ├── splits/
    │   ├── train.csv
    │   └── validation.csv
    └── lits_train.csv
```

**Note:** The dataset is not included in this GitHub repository. You must obtain the dataset separately and organize it according to the structure above.

## 4. Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Streamlit
- Matplotlib
- U-Net architecture

## 5. Model Architecture

The project uses a U-Net convolutional neural network for image segmentation.

The model has:

- An encoder that extracts image features.
- A bottleneck that learns deeper representations.
- A decoder that reconstructs spatial information.
- Skip connections that combine encoder and decoder features.
- Separate output heads for liver and tumor segmentation.

## 6. Evaluation Metrics

The model can be evaluated using the following segmentation metrics:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall

Tumor segmentation is particularly challenging because tumor pixels occupy a relatively small portion of many CT slices.

## 7. Project Structure

```text
liver-image-segmentation/
├── src/
│   └── segmentation/
│       ├── app.py
│       ├── evaluate.py
│       ├── lits_dataset.py
│       ├── losses.py
│       ├── metrics.py
│       ├── prepare_split.py
│       ├── train.py
│       └── unet.py
├── check_tumor_masks.py
├── compare_tumor_labels.py
├── count_positive_tumor_masks.py
├── find_demo_image.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/srinithi19092005/liver-image-segmentation.git
cd liver-image-segmentation
```

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Obtain the LiTS dataset separately and organize the images, liver masks, tumor masks, and CSV split files according to the dataset structure described above.

## 9. Training

After preparing the dataset, run:

```powershell
python -m src.segmentation.train
```

Training settings and data paths should match the configuration in the training script.

## 10. Evaluation

To evaluate a trained model, run:

```powershell
python -m src.segmentation.evaluate
```

Ensure that the required trained checkpoint and validation data are available before running evaluation.

## 11. Streamlit Application

Start the web application using:

```powershell
streamlit run src/segmentation/app.py
```

The application is intended to demonstrate liver and tumor segmentation predictions on uploaded CT images.

## 12. Limitations

- Model performance depends on the quality and diversity of the training data.
- Tumor segmentation is challenging because tumors may occupy very small regions in CT images.
- The project is an educational and research prototype.
- The predictions have not been established as suitable for clinical diagnosis or treatment decisions.

## 13. Future Improvements

Potential improvements include:

- Training on a larger and more representative dataset.
- Improving tumor segmentation performance.
- Adding more visualization and evaluation features.
- Comparing U-Net with other segmentation architectures.
- Improving the Streamlit interface.

## 14. Author

**GitHub:** [srinithi19092005](https://github.com/srinithi19092005)\# Liver and Tumor Segmentation Using U-Net



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


