#  Experiment Tracking Labs – CIFAR-10 (W&B + Keras)

This lab demonstrates **experiment tracking**, **metric logging**, and **model management** using **Weights & Biases (W&B)** on the **CIFAR-10 dataset**.
It covers dataset preparation, model building, callbacks, logging, model saving, and experiment comparison.

##  **Objectives**

* Track ML experiments using **Weights & Biases**
* Log:

  * Training & validation metrics
  * Learning rate
  * Sample predictions
  * Confusion matrix
* Save trained models locally
* Compare runs visually in W&B dashboard
* Understand how experiment tracking fits into the **MLOps lifecycle**


##  **Folder Structure**

```
Experiment_Tracking_Labs/
│── Lab2.ipynb
│── cifar10_trainer.py (or your .py file)
│── outputs/
│      └── cifar10_cnn.keras
│── README.md
```


##  **Dataset: CIFAR-10**

* 60,000 images (32×32 RGB)
* 10 classes:
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`
* Train: 50,000 images
* Test: 10,000 images
* Normalized to `[0, 1]`
* One-hot encoded labels

##  **Model Architecture**

A simple Convolutional Neural Network (CNN):

* Conv2D → ReLU
* MaxPooling
* Conv2D → ReLU
* MaxPooling
* Flatten
* Dense(128) → ReLU
* Dropout
* Dense(10) → Softmax

Optimizer: **Adam**
Loss: **categorical_crossentropy**
Batch size: **32**
Epochs: **5** (default)


##  **Weights & Biases Logging**

The lab uses custom callbacks to log:

###  Training Metrics

* loss
* accuracy
* val_loss
* val_accuracy

###  Learning Rate

Custom callback logs dynamic LR each epoch.

###  Sample Predictions

A W&B Table with:

* Images
* True labels
* Predicted labels

###  Confusion Matrix

Logged at the end of every epoch.

###  Saved Model Path

Model saved as:

```
cifar10_cnn.keras
```

and path logged to W&B.


##  **How to Run**

### 1. Install dependencies

```bash
pip install tensorflow keras wandb numpy matplotlib
```

### 2. Login to W&B

```bash
wandb login
```

### 3. Run the notebook or trainer

```python
from cifar10_trainer import CIFAR10Trainer

trainer = CIFAR10Trainer()
trainer.train()
```

##  **Results Obtained**

| Metric              | Value        |
| ------------------- | ------------ |
| Training Accuracy   | ~0.55 – 0.60 |
| Validation Accuracy | ~0.58 – 0.63 |
| Training Loss       | ~1.20        |
| Validation Loss     | ~1.13        |

This is **expected** for:

* A small CNN
* 5 epochs
* CIFAR-10
* No data augmentation

##  **Why Validation Accuracy < 70%?**

CIFAR-10 is harder than MNIST/Fashion-MNIST.
A simple CNN trained for only 5 epochs normally achieves **55–65%**.

Higher accuracy requires:

* Data augmentation
* Deeper CNN
* LR scheduling
* More epochs

I can generate an improved version if you need.

##  **What This Lab Demonstrates (Key Talking Points)**

When explaining to your TA:

* Implemented end-to-end experiment tracking using **W&B**
* Switched dataset from Fashion-MNIST → **CIFAR-10**
* Used callbacks to log:

  * Metrics
  * LR
  * Predictions
  * Confusion matrix
* Saved model locally and logged path
* Handled Windows symlink restrictions (replaced `wandb.save()` with JSON logging)
* Ensured reproducible runs & comparison

##  **Conclusion**

This lab successfully shows how to integrate experiment tracking into an ML workflow using W&B.
It highlights the importance of:

* Tracking metrics
* Visualizing model performance
* Maintaining experiment history
* Saving models for reproducibility

