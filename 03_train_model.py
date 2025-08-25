import pandas as pd
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Path

FEATURES_PATH = os.path.join('features', 'resnet50_features.csv') 
MODELS_PATH = 'models'

def train_and_evaluate():
    """Loads features, trains classifiers, and saves the best model."""
    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH)

    # Handle potential NaN/infinity values from feature extraction if any 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Defining models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', C=0.1),
        'SVC': SVC(kernel='rbf', C=1, probability=True), 
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }

    best_model = None
    best_score = 0.0

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Create a pipelining  scaler and the model
    
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        print(f"Classification Report for {name}:")
        report = classification_report(y_test, y_pred)
        print(report)
        
        # Checking  for the best model based on weighted F1-score
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        current_score = report_dict['weighted avg']['f1-score']
        
        if current_score > best_score:
            best_score = current_score
            best_model = pipeline
            print(f"*** New best model found: {name} with F1-score: {best_score:.4f} ***")

    # output - best model
    if best_model:
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        
        model_filename = os.path.join(MODELS_PATH, 'best_oct_resnet_classifier.pkl')
        joblib.dump(best_model, model_filename)
        print(f"\nBest model saved to '{model_filename}'")

if __name__ == '__main__':
    train_and_evaluate()
