import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --- Configuration ---
DATA_PATH = os.path.join('data', 'OCT2017', 'train')
OUTPUT_PATH = 'features'
# ResNet50 uses 224x224 images as input
IMAGE_SIZE = (224, 224) 
CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# ---------------------

def get_feature_extractor():
    """Creates and returns the ResNet50 model for feature extraction."""
    # Load ResNet50 base model, pre-trained on ImageNet
    # include_top=False means we don't include the final classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Add a global average pooling layer to get a fixed-size feature vector
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # The final model for feature extraction
    model = Model(inputs=base_model.input, outputs=x)
    return model

def extract_features(image_path, model):
    """Reads an image, preprocesses it for ResNet50, and extracts features."""
    try:
        # Load and resize the image
        img = load_img(image_path, target_size=IMAGE_SIZE)
        
        # Convert the image to a numpy array
        img_array = img_to_array(img)
        
        # Expand dimensions to create a "batch" of 1 image
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Preprocess the image for the ResNet50 model (scaling, etc.)
        img_preprocessed = preprocess_input(img_batch)
        
        # Extract features
        features = model.predict(img_preprocessed, verbose=0)
        
        # Flatten the features to a 1D array
        return features.flatten()
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_and_save_features():
    """Processes all images in the dataset and saves the features to a CSV."""
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Get the feature extraction model
    print("Loading ResNet50 model...")
    model = get_feature_extractor()
    print("Model loaded.")

    all_features = []
    all_labels = []

    print("Starting feature extraction...")
    for class_name in CLASSES:
        print(f"Processing class: {class_name}")
        class_path = os.path.join(DATA_PATH, class_name)
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        for image_path in tqdm(image_files, desc=f"Extracting {class_name} features"):
            features = extract_features(image_path, model)
            if features is not None:
                all_features.append(features)
                all_labels.append(class_name)

    print("Creating DataFrame...")
    feature_df = pd.DataFrame(all_features)
    feature_df['label'] = all_labels

    # Save to a CSV file
    output_file = os.path.join(OUTPUT_PATH, 'resnet50_features.csv')
    feature_df.to_csv(output_file, index=False)
    print(f"Features saved successfully to '{output_file}'!")

if __name__ == '__main__':
    # Ensure TensorFlow is not using all GPU memory if you have other processes running
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    process_and_save_features()