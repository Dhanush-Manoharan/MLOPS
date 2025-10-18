from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

app = Flask(__name__, static_folder='statics')

# Load the trained model
print("Loading model...")
model = keras.models.load_model('penguin_model.keras')
print("Model loaded successfully!")

# Load scaler and label encoder
print("Loading preprocessing objects...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print(f"Classes: {label_encoder.classes_}")
print("Ready to make predictions!")

@app.route('/')
def home():
    return "Welcome to the Penguin Species Classification API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data
            bill_length = float(request.form['bill_length'])
            bill_depth = float(request.form['bill_depth'])
            flipper_length = float(request.form['flipper_length'])
            body_mass = float(request.form['body_mass'])
            
            # Prepare input array
            input_data = np.array([[bill_length, bill_depth, flipper_length, body_mass]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled, verbose=0)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))
            
            # Get species name
            species = label_encoder.inverse_transform([predicted_class])[0]
            
            return jsonify({
                "species": species,
                "confidence": confidence,
                "probabilities": {
                    label_encoder.classes_[0]: float(prediction[0][0]),
                    label_encoder.classes_[1]: float(prediction[0][1]),
                    label_encoder.classes_[2]: float(prediction[0][2])
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        # GET request - return the HTML form
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)