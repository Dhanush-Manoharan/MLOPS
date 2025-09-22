#  Penguin Species Classification API

**Course**: IE 7374 MLOps  
**Student**: Dhanush Manoharan  
**Professor**: Ramin Mohammadi  
**Semester**: Fall 2024  
**Project**: FastAPI Lab 1 - Machine Learning Model Deployment

##  Project Overview

This project implements a production-ready REST API for classifying Palmer Archipelago penguin species using machine learning. The API predicts whether a penguin belongs to Adelie, Chinstrap, or Gentoo species based on physical measurements and location data.

##  Features

- **RESTful API** built with FastAPI framework
- **Machine Learning Model**: Random Forest Classifier with ~97% accuracy
- **Automatic Data Validation**: Input validation using Pydantic models
- **Missing Value Handling**: Automatic imputation for incomplete data
- **Interactive Documentation**: Swagger UI and ReDoc interfaces
- **Model Training Endpoint**: Retrain models via API
- **Performance Metrics**: Track model accuracy and feature importance
- **CORS Support**: Cross-origin resource sharing enabled

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

**1. Clone the repository:**

git clone https://github.com/Dhanush-Manoharan/MLOPS.git
cd MLOPS/fastapi_lab1 


**2. Install dependencies:**
  pip install -r requirements.txt

**3. Train the model (first time only):**
  cd src
  python train.py

**4. Start the Server:**
  uvicorn main:app --reload

**5. Access the API:**
    http://localhost:8000/docs#/

   
** Project Structure**
fastapi_lab1/
│
├── src/                      # Source code
│   ├── main.py              # FastAPI application and endpoints
│   ├── data.py              # Data loading and preprocessing
│   ├── train.py             # Model training pipeline
│   └── predict.py           # Prediction logic
│
├── model/                    # Saved models and encoders
│   ├── penguin_model.pkl    # Trained Random Forest model
│   ├── scaler.pkl           # Feature scaler
│   ├── species_encoder.pkl  # Label encoder for species
│   └── metrics.json         # Model performance metrics
│
├── assets/                   # Screenshots and documentation
├── requirements.txt          # Python dependencies
├── penguins.csv             # Dataset
└── README.md                # This file

**Model Information**
Algorithm

Random Forest Classifier with 150 estimators
Cross-validated for optimal hyperparameters
Handles imbalanced classes with stratified splitting

Features Used

Bill Length (mm) - Length of penguin's bill
Bill Depth (mm) - Depth of penguin's bill
Flipper Length (mm) - Length of penguin's flipper
Body Mass (g) - Weight of penguin
Island - Location (Biscoe, Dream, or Torgersen)
Sex - Male, Female, or Unknown
Year - Year of observation

Performance Metrics

Accuracy: ~97%
Precision: ~97%
Recall: ~97%
F1-Score: ~97%

Example Usage
Making a Prediction
Request:
POST /predict
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "island": "Torgersen",
  "sex": "male",
  "year": 2007
}

Response:
{
  "prediction": 0,
  "species": "Adelie",
  "confidence": 0.9967,
  "probabilities": {
    "Adelie": 0.9967,
    "Chinstrap": 0.0033,
    "Gentoo": 0.0
  },
  "confidence_level": "High",
  "fun_fact": "Adelie penguins are the smallest species in Antarctica!"
}

API Screenshots
Screenshots demonstrating the working API can be found in the /assets folder:

Swagger UI interface
Successful prediction example
Model training endpoint
API health check

 Technologies Used

FastAPI: Modern web framework for building APIs
scikit-learn: Machine learning library
Pandas: Data manipulation and analysis
NumPy: Numerical computing
Uvicorn: ASGI server
Pydantic: Data validation

 Model Training
To retrain the model with new data:

Via API:

   POST http://localhost:8000/train

Via Command Line:

bash   cd src
   python train.py

 Learning Outcomes
This project demonstrates:

RESTful API design principles
Machine learning model deployment
Data validation and error handling
API documentation with OpenAPI/Swagger
Model versioning and persistence
Production-ready code organization

 Acknowledgments

Palmer Archipelago Penguin Dataset from Dr. Kristen Gorman
Professor Ramin Mohammadi for course guidance
FastAPI documentation and community

 License
This project is submitted as part of IE 7374 MLOps coursework.


For questions or issues, please contact: manoharan.d@northeastern.edu
