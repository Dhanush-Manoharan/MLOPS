import numpy as np
import joblib
from typing import Dict
import os
import warnings
warnings.filterwarnings('ignore')

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, 'model')
    
    model = joblib.load(os.path.join(model_dir, 'penguin_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    species_encoder = joblib.load(os.path.join(model_dir, 'species_encoder.pkl'))
    numerical_imputer = joblib.load(os.path.join(model_dir, 'numerical_imputer.pkl'))
    island_encoder = joblib.load(os.path.join(model_dir, 'island_encoder.pkl'))
    sex_encoder = joblib.load(os.path.join(model_dir, 'sex_encoder.pkl'))
    
    return model, scaler, species_encoder, numerical_imputer, island_encoder, sex_encoder

def make_prediction(features: Dict) -> Dict:
    model, scaler, species_encoder, numerical_imputer, island_encoder, sex_encoder = load_models()
    
    numerical_values = []
    numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    for feat in numerical_features:
        val = features.get(feat, np.nan)
        if val is None or val == '':
            val = np.nan
        numerical_values.append(float(val) if not isinstance(val, float) else val)
    
    numerical_array = np.array([numerical_values])
    numerical_imputed = numerical_imputer.transform(numerical_array)[0]
    
    island = features.get('island', 'Biscoe')
    try:
        island_encoded = island_encoder.transform([island])[0]
    except:
        island_encoded = 0
        island = 'Biscoe'
    
    sex = features.get('sex', 'unknown')
    if sex is None or sex == '':
        sex = 'unknown'
    try:
        sex_encoded = sex_encoder.transform([sex])[0]
    except:
        sex_encoded = sex_encoder.transform(['unknown'])[0]
        sex = 'unknown'
    
    year = features.get('year', 2008)
    if year is None or year == '':
        year = 2008
    
    feature_array = np.array([[
        numerical_imputed[0], numerical_imputed[1], numerical_imputed[2], numerical_imputed[3],
        island_encoded, sex_encoded, year
    ]])
    
    features_scaled = scaler.transform(feature_array)
    
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    species = species_encoder.inverse_transform([prediction])[0]
    
    species_probs = {}
    for i, sp in enumerate(species_encoder.classes_):
        species_probs[sp] = float(probabilities[i])
    
    confidence = float(max(probabilities))
    
    # Species-specific fun facts
    fun_facts = {
        'Adelie': 'Adelie penguins are the smallest species in Antarctica and excellent swimmers!',
        'Chinstrap': 'Chinstrap penguins get their name from the thin black band under their heads!',
        'Gentoo': 'Gentoo penguins are the fastest underwater swimmers of all penguins, reaching 36 km/h!'
    }
    
    result = {
        'prediction': int(prediction),
        'species': species,
        'confidence': confidence,
        'probabilities': species_probs,
        'confidence_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
        'fun_fact': fun_facts.get(species, f'{species} penguin detected!'),
        'species_info': {
            'name': species,
            'confidence_score': f'{confidence:.2%}',
            'all_probabilities': species_probs
        },
        'input_summary': {
            'measurements': {
                'bill_length': f"{numerical_imputed[0]:.1f} mm",
                'bill_depth': f"{numerical_imputed[1]:.1f} mm",
                'flipper_length': f"{numerical_imputed[2]:.1f} mm",
                'body_mass': f"{numerical_imputed[3]:.0f} g"
            },
            'location': island,
            'sex': sex,
            'year': year
        },
        'data_quality': {
            'missing_values_handled': True,
            'imputation_used': any(np.isnan(numerical_values))
        }
    }
    
    return result

if __name__ == "__main__":
    test_data = {
        'bill_length_mm': 39.1,
        'bill_depth_mm': 18.7,
        'flipper_length_mm': 181.0,
        'body_mass_g': 3750.0,
        'island': 'Torgersen',
        'sex': 'male',
        'year': 2007
    }
    
    result = make_prediction(test_data)
    print(f"Prediction: {result['species']} (Confidence: {result['confidence']:.2%})")