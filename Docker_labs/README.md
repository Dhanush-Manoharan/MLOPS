
# Docker Lab 2 - Penguin Species Classification API

A machine learning web application that classifies penguin species based on physical measurements, built with TensorFlow, Flask, and Docker using multi-stage builds.

##  About the Project

This project implements a neural network classifier to predict penguin species (Adelie, Chinstrap, or Gentoo) based on four physical measurements:
- Bill Length (mm)
- Bill Depth (mm)
- Flipper Length (mm)
- Body Mass (g)

The application uses a **multi-stage Docker build** to:
1. Train the model in the first stage
2. Serve predictions via Flask API in the second stage

##  Features

- **Multi-stage Docker build** for efficient containerization
- **Neural network model** with 95%+ accuracy
- **Interactive web interface** for easy predictions
- **RESTful API** for programmatic access
- **Real-time predictions** with confidence scores

##  Technologies Used

- **Python 3.9**
- **TensorFlow 2.15** - Deep learning framework
- **Flask** - Web framework
- **scikit-learn** - Data preprocessing
- **Docker** - Containerization
- **Seaborn** - Penguins dataset

##  Project Structure
```
Docker_labs/
├── src/
│   ├── model_training.py       # Neural network training script
│   ├── main.py                 # Flask API server
│   ├── templates/
│   │   └── predict.html        # Web interface
│   └── statics/                # Screenshots and images
├── Dockerfile                  # Multi-stage Docker build
├── requirements.txt            # Python dependencies
└── HOWTO                       # Build and run commands
```

##  How to Run

### Prerequisites
- Docker Desktop installed and running

### Build the Docker Image
```bash
docker build -t penguin-classifier .
```

This command will:
- Download Python 3.9 base image
- Install all dependencies
- Train the neural network model
- Create the serving container

### Run the Container
```bash
docker run -p 4000:4000 penguin-classifier
```

### Access the Application

Open your browser and go to:
```
http://localhost:4000/predict
```

##  Example Test Cases

### Adelie Penguin
- Bill Length: 39.1 mm
- Bill Depth: 18.7 mm
- Flipper Length: 181 mm
- Body Mass: 3750 g

### Chinstrap Penguin
- Bill Length: 46.5 mm
- Bill Depth: 17.9 mm
- Flipper Length: 192 mm
- Body Mass: 3500 g

### Gentoo Penguin
- Bill Length: 47.5 mm
- Bill Depth: 15.0 mm
- Flipper Length: 215 mm
- Body Mass: 4975 g

##  Model Performance

- **Architecture**: 3-layer Neural Network
- **Training Epochs**: 150
- **Test Accuracy**: ~95-98%
- **Dataset**: Palmer Penguins (seaborn)

##  Docker Multi-Stage Build

The Dockerfile uses two stages:

**Stage 1 (Training):**
- Trains the model using `model_training.py`
- Saves `penguin_model.keras`, `scaler.pkl`, and `label_encoder.pkl`

**Stage 2 (Serving):**
- Copies trained model from Stage 1
- Sets up Flask web server
- Exposes port 4000
- Runs the prediction API

##  API Endpoints

### `GET /`
Returns welcome message

### `GET /predict`
Returns the web interface

### `POST /predict`
Accepts form data with penguin measurements and returns:
```json
{
  "species": "Adelie",
  "confidence": 0.96,
  "probabilities": {
    "Adelie": 0.96,
    "Chinstrap": 0.03,
    "Gentoo": 0.01
  }
}
```

##  Stopping the Container

Press `Ctrl + C` in the terminal where the container is running.
