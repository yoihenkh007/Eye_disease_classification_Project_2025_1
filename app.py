import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array


CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# path to best_model
MODEL_FILENAME = 'best_oct_resnet_classifier.pkl' 

# models path
MODEL_PATH = os.path.join('models', MODEL_FILENAME)
# RESIZE ti fit model 
IMAGE_SIZE = (224, 224) 
# ----------------------------------------------------------------

# SIDEBAR with info
with st.sidebar:
    st.title("About the OCT Classifier")
    st.markdown("""
        This application uses a powerful Deep Learning model (ResNet50) to classify eye diseases from Optical Coherence Tomography (OCT) images.
    """)

    st.header("Pipeline Overview")
    st.markdown("""
        1.  **Image Input:** An OCT image is uploaded.
        2.  **Preprocessing:** The image is resized to a standard size (224x224).
        3.  **Feature Extraction:** The image is passed through a pre-trained **ResNet50** model. This convolutional neural network, originally trained on millions of images, acts as a sophisticated feature extractor to capture complex patterns, textures, and shapes. It transforms the image into a high-dimensional numerical vector.
        4.  **Classification:** The feature vector is fed into a pre-trained classic Machine Learning model (like Random Forest or SVC) which predicts the final diagnosis.
    """)

    st.header("Algorithms Used")
    st.markdown("""
        - **Feature Extractor:** ResNet50 (pre-trained on ImageNet)
        - **Classifier:** The final model is a scikit-learn pipeline, likely containing a Random Forest or Support Vector Classifier.
    """)
    st.info(f"The currently loaded model is **'{MODEL_FILENAME}'**.")


@st.cache_resource # Caches the model for faster re-runs
def load_classifier():
    """Loads the saved scikit-learn pipeline."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Classifier file not found at {MODEL_PATH}. Please run the training script first.")
        return None

@st.cache_resource # Cache the feature extractor model
def load_feature_extractor():
    """Creates and returns the ResNet50 model for feature extraction."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def extract_features(image_np, model):
    """Extracts ResNet50 features from a raw image."""
    # Ensure image is 3-channel (RGB)
    if len(image_np.shape) == 2: # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4: # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
    img_resized = cv2.resize(image_np, IMAGE_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    features = model.predict(img_preprocessed, verbose=0)
    return features.flatten()

# --- Main App UI ---
st.title("👁️ OCT Eye Disease Classifier (ResNet50-Powered)")
st.write(f"Upload an OCT scan to predict the diagnosis from the following categories: **{', '.join(CLASSES)}**.")

classifier_model = load_classifier()
feature_extractor_model = load_feature_extractor()

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None and classifier_model is not None and feature_extractor_model is not None:
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    image_np = np.array(image)
    
    # Feature Extraction - ResNet50 model
    features = extract_features(image_np, feature_extractor_model)
    features_reshaped = features.reshape(1, -1)
    
    # fetching prediction 
    prediction_proba = classifier_model.predict_proba(features_reshaped)
    prediction = classifier_model.predict(features_reshaped)[0]
    
    st.success(f"**Predicted Diagnosis: {prediction}**")
    
    st.header("Prediction Confidence")
    confidence_scores = prediction_proba[0]
    confidence_df = pd.DataFrame({
        'Class': classifier_model.classes_,
        'Confidence (%)': [f"{score*100:.2f}%" for score in confidence_scores]
    })
    confidence_df = confidence_df.sort_values(by='Confidence (%)', ascending=False).reset_index(drop=True)
    st.dataframe(confidence_df, use_container_width=True, hide_index=True)
    
    st.header("Confidence Bar Chart")
    confidence_graph_df = pd.DataFrame(prediction_proba, columns=classifier_model.classes_).T
    confidence_graph_df.rename(columns={0: 'Confidence'}, inplace=True)
    st.bar_chart(confidence_graph_df)

elif classifier_model is None:
    st.warning("Classifier model not loaded. Please ensure the model file exists and the training script has been run.")