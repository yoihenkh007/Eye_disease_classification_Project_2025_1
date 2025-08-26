
# OCT Eye Disease Classification using Transfer Learning and Scikit-learn

This project demonstrates a complete machine learning pipeline for classifying eye diseases from Optical Coherence Tomography (OCT) images. It utilizes a hybrid approach, leveraging a pre-trained deep learning model (ResNet50) for powerful feature extraction and classic machine learning classifiers (e.g., Random Forest, SVC) for efficient and accurate classification. The project culminates in an interactive web application built with Streamlit for real-time predictions.

## 📋 Project Overview

The goal of this project is to accurately classify OCT scans into one of four categories:

  * **CNV** (Choroidal Neovascularization)
  * **DME** (Diabetic Macular Edema)
  * **DRUSEN**
  * **NORMAL**

The workflow is designed to be modular and reproducible, broken down into distinct steps: data analysis, feature extraction, model training, evaluation, and deployment as a local web app.

## ✨ Key Features

  * **Modular Pipeline**: The project is organized into a series of numbered Python scripts and notebooks, making the workflow easy to understand and execute.
  * **Deep Learning Feature Extraction**: Uses the ResNet50 model, pre-trained on ImageNet, to extract high-level, discriminative features from the OCT images.
  * **Classic Machine Learning Classification**: Employs robust and efficient Scikit-learn classifiers to train on the extracted features, demonstrating a powerful hybrid approach.
  * **Comprehensive Analysis**: Includes Jupyter notebooks for exploratory data analysis (EDA) and in-depth model performance evaluation.
  * **Interactive Web Application**: A user-friendly Streamlit application allows for easy testing by uploading an OCT image and receiving an instant diagnosis prediction.

## 💾 Dataset

This project uses the **"OCT2017"** dataset, which contains over 84,000 OCT images of retinal tissue. The dataset is organized by the four categories mentioned above.
This dataset has 4 classes as disease
-- Current model used only 42,976 images
  Training- 3000 each classes
  Testing- 242 each class

You can download the dataset from Kaggle: [Kermany et al., OCT2017 Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

## 📂 Project Structure

```
.
├── data/
│   └── OCT2017/
│       └── train/
│           ├── CNV/
│           ├── DME/
│           ├── DRUSEN/
│           └── NORMAL/
├── features/
│   └── resnet50_features.csv
├── models/
│   └── best_oct_resnet_classifier.pkl
├── .gitignore
├── data_visualization.ipynb
├── 02_extract_features.py
├── 03_train_model.py
├── 04_model_evaluation.ipynb
├── app.py
├── README.md
└── requirements.txt
```

## 🚀 Setup and Installation

Follow these steps to set up the project environment.

### - Prerequisites

  * Python 3.9 or higher


###  Download and Place the Dataset

1.  Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) provided above.
2.  Unzip the file.
3.  Place the `train` folder inside the `data/OCT2017/` directory as shown in the project structure. Ensure the final path is `data/OCT2017/train/`.

## ⚙️ Workflow and How to Run

Execute the scripts in the following order.

### 1\. Extract Features

This script will process all images in the `data` directory, use ResNet50 to extract features, and save them to a CSV file.
**This will take a significant amount of time and CPU/GPU resources.**

```bash
python 02_extract_features.py
```

  * **Output**: A file named `resnet50_features.csv` will be created in the `features/` directory.

### 2\. Train the Classifier

This script loads the extracted features, trains several Scikit-learn models (Logistic Regression, SVC, Random Forest), evaluates them, and saves the best-performing model.

```bash
python 03_train_model.py
```

  * **Output**: The best model pipeline will be saved as `best_oct_resnet_classifier.pkl` in the `models/` directory.

### 3\. Evaluate the Best Model

Run the Jupyter Notebook `data_visualization.ipynbb` to perform a detailed analysis of the best model's performance, including a classification report and a confusion matrix.

```bash
jupyter notebook data_visualization.ipynb
```

### 4\. Run the Interactive Web App

Launch the Streamlit application to perform live predictions.
`app.py`



  * Open your web browser and navigate to the local URL provided in the terminal to interact with the application.

## 🛠️ Technology Stack

  * **Python**: Core programming language.
  * **TensorFlow / Keras**: For loading the pre-trained ResNet50 model for feature extraction.
  * **Scikit-learn**: For training and evaluating the classification models.
  * **Pandas**: For data manipulation and handling the feature set.
  * **Streamlit**: For building and serving the interactive web application.
  * **Matplotlib & Seaborn**: For data visualization and model evaluation plots.
  * **Pillow & OpenCV**: For image processing.

## 📈 Model Performance

The hybrid model combining ResNet50 features with a Random Forest classifier achieves excellent performance, with a weighted F1-score of approximately **97%** on the test set. The detailed per-class metrics can be reviewed in the `data_visualization.ipynb` notebook.


