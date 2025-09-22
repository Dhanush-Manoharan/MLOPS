import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

def find_penguins_csv():
    """Find penguins.csv in various possible locations"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    possible_paths = [
        os.path.join(project_root, 'penguins.csv'),
        os.path.join(current_dir, '..', 'penguins.csv'),
        os.path.join(current_dir, 'penguins.csv'),
        'C:/SUBS/mlops_labs/penguins.csv',
        'C:/SUBS/mlops_labs/fastapi_lab1/penguins.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(
        f"penguins.csv not found. Searched in:\n" + 
        "\n".join(possible_paths)
    )

def load_data():
    """Load and preprocess the Penguins dataset with missing value handling"""
    file_path = find_penguins_csv()
    print(f"Loading data from: {file_path}")
    
    df = pd.read_csv(file_path)
    
    print(f"Initial dataset shape: {df.shape}")
    
    missing_counts = df.isnull().sum()
    print(f"Total missing values: {missing_counts.sum()}")
    
    df = df.dropna(subset=['species'])
    df['sex'] = df['sex'].fillna('unknown')
    
    target_column = 'species'
    y = df[target_column].values
    
    species_encoder = LabelEncoder()
    y = species_encoder.fit_transform(y)
    
    numerical_features = ['bill_length_mm', 'bill_depth_mm', 
                         'flipper_length_mm', 'body_mass_g']
    
    X_numerical = df[numerical_features].copy()
    
    numerical_imputer = SimpleImputer(strategy='median')
    X_numerical_imputed = numerical_imputer.fit_transform(X_numerical)
    
    island_encoder = LabelEncoder()
    island_encoded = island_encoder.fit_transform(df['island'])
    
    sex_encoder = LabelEncoder()
    sex_encoded = sex_encoder.fit_transform(df['sex'])
    
    year_values = df['year'].values
    
    X = np.column_stack([X_numerical_imputed, island_encoded, sex_encoded, year_values])
    
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(numerical_imputer, os.path.join(model_dir, 'numerical_imputer.pkl'))
    joblib.dump(island_encoder, os.path.join(model_dir, 'island_encoder.pkl'))
    joblib.dump(sex_encoder, os.path.join(model_dir, 'sex_encoder.pkl'))
    joblib.dump(species_encoder, os.path.join(model_dir, 'species_encoder.pkl'))
    
    feature_names = numerical_features + ['island_encoded', 'sex_encoded', 'year']
    target_names = species_encoder.classes_.tolist()
    
    print(f"Final shape: X={X.shape}, y={y.shape}")
    print(f"Encoders saved to: {model_dir}")
    
    return X, y, feature_names, target_names

def preprocess_data(X_train, X_test):
    """Scale the features for better model performance"""
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"Scaler saved to: {model_dir}")
    
    return X_train_scaled, X_test_scaled

def split_data(X, y):
    """Split data into training and testing sets"""
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        raise ValueError(f"Not enough samples ({len(X)})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_feature_names():
    return ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
            'body_mass_g', 'island_encoded', 'sex_encoded', 'year']

def get_species_names():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_dir = os.path.join(project_root, 'model')
        species_encoder = joblib.load(os.path.join(model_dir, 'species_encoder.pkl'))
        return species_encoder.classes_.tolist()
    except:
        return ['Adelie', 'Chinstrap', 'Gentoo']

if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    print(f"Data loaded: {len(X)} samples")