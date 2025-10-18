# Stage 1: Model Training
FROM python:3.9 AS model_training

WORKDIR /app

# Copy the source code for model training
COPY src/model_training.py /app/

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Run the model training script
RUN python model_training.py

# Stage 2: Serve Predictions
FROM python:3.9 AS serving

WORKDIR /app

# Copy the trained model and preprocessing objects from the previous stage
# Copy the trained model and preprocessing objects from the previous stage
COPY --from=model_training /app/penguin_model.keras /app/
COPY --from=model_training /app/scaler.pkl /app/
COPY --from=model_training /app/label_encoder.pkl /app/

# Copy the source code for serving predictions
COPY src/main.py /app/

# Copy the templates directory
COPY src/templates /app/templates

# Copy the statics directory
COPY src/statics /app/statics

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Expose port 4000
EXPOSE 4000

# Define environmental variables
ENV NAME World

# Run the Flask application for serving predictions
CMD ["python", "main.py"]