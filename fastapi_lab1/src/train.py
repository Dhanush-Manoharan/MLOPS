import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import data functions - handle both direct run and import scenarios
try:
    from data import get_feature_names
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from data import get_feature_names

def train_multiple_models(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = None
    
    print("\nComparing models...")
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
            print(f"{name}: Accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"Failed to train {name}: {e}")
    
    return best_model, best_model_name, results

def train_best_model(X_train, y_train, X_test, y_test, model_name="Random Forest"):
    print(f"\nTraining final model: {model_name}")
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'training_date': datetime.now().isoformat()
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'penguin_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    if hasattr(model, 'feature_importances_'):
        feature_names = get_feature_names()
        
        importance_dict = {
            'features': feature_names,
            'importance': model.feature_importances_.tolist()
        }
        
        importance_path = os.path.join(model_dir, 'feature_importance.json')
        with open(importance_path, 'w') as f:
            json.dump(importance_dict, f, indent=4)
    
    return model, metrics

if __name__ == "__main__":
    from data import load_data, split_data, preprocess_data
    
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    best_model, best_name, results = train_multiple_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    final_model, final_metrics = train_best_model(
        X_train_scaled, y_train, X_test_scaled, y_test, best_name
    )
    
    print("\nTraining complete!")