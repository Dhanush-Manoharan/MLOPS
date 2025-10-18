import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Loading Penguins dataset...")
# Load penguins dataset from seaborn
penguins = sns.load_dataset('penguins')

# Drop rows with missing values
penguins = penguins.dropna()

print(f"Dataset shape: {penguins.shape}")
print(f"Species: {penguins['species'].unique()}")

# Prepare features and target
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = penguins['species']

# Encode species to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Classes: {label_encoder.classes_}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Building model...")
# Build neural network
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Train
print("Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# Save model and preprocessing objects
print("Saving model and preprocessing objects...")
model.save('penguin_model.keras')

import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model saved successfully!")
print(f"Classes: {list(label_encoder.classes_)}")