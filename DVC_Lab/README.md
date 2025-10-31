### Data Version Control (DVC) Lab — Google Cloud Integration

###  Objective

The goal of this lab is to implement **Data Version Control (DVC)** integrated with **Google Cloud Storage** to manage dataset versioning efficiently.
This lab demonstrates how to:

* Track data changes using DVC
* Store and retrieve dataset versions remotely on Google Cloud
* Ensure reproducibility in machine learning workflows


###  Tools & Technologies Used

* **Python 3.11+**
* **VS Code**
* **DVC with Google Cloud Support** (`dvc[gs]`)
* **Git & GitHub**
* **Google Cloud Storage (GCS)**
* **Virtual Environment (venv)**



###  Setup & Implementation Steps

#### 1️ Environment Setup

```bash
# Create and activate virtual environment
python -m venv dvc
dvc\Scripts\activate

# Install required dependencies
pip install "dvc[gs]"
```

####  Initialize Git & DVC

```bash
git init
dvc init
```

This creates the `.dvc/` folder and config files needed for version tracking.

####  Add Dataset and Enable Tracking

Dataset used: **Iris Dataset**

```bash
mkdir data
# Download dataset (or use your own)
# Example: iris.csv
dvc add data/iris.csv
git add data/iris.csv.dvc .gitignore
git commit -m "Track iris dataset with DVC"
```

####  Connect DVC to Google Cloud Bucket

```bash
# Add Google Cloud bucket as remote storage
dvc remote add -d myremote gs://dhanushlab

# Authenticate with service account key
dvc remote modify myremote credentialpath "C:\Users\dhanu\OneDrive\Documents\mlopslab-476800-7644e65fe0c6.json"

# Commit configuration
git add .dvc/config
git commit -m "Configured DVC remote with Google Cloud bucket"
```

####  Push Data to Cloud

```bash
dvc push
```

This uploads the tracked dataset to your **Google Cloud Storage bucket (`dhanushlab`)**.
You can verify this in the **GCP Console → Storage → dhanushlab → Objects tab**.

####  Create a New Dataset Version

To demonstrate version control, modify the dataset (e.g., remove a few rows) and save it as `iris_v2.txt`.

```bash
dvc add data/iris_v2.txt
git add data/iris_v2.txt.dvc
git commit -m "Added iris_v2.txt - new dataset version"
dvc push
```

 DVC detects changes and uploads a new hash file to the cloud, maintaining both dataset versions.

####  Revert to Previous Version

```bash
git log --oneline
git checkout <old_commit_hash>
dvc checkout
```

This restores the old dataset version from the cloud using the stored hash reference.

###  Folder Structure

```
DVC_Lab/
│
├── data/
│   ├── iris.csv
│   ├── iris.csv.dvc
│   ├── iris_v2.txt
│   └── iris_v2.txt.dvc
│
├── result/
│   ├── Bucket_Monitoring.png
│   └── Cloud_storage_result.png
│
├── .dvc/
├── requirements.txt
└── README.md
```

###  Results & Verification

* **Bucket_Monitoring.png** – confirms bucket structure and successful upload.
* **Cloud_storage_result.png** – shows multiple hashed dataset versions (`01`, `cb` folders) in Google Cloud Storage.

###  Conclusion

This lab successfully demonstrated:

* Data versioning using **DVC**
* Remote storage integration with **Google Cloud**
* Version tracking, updating, and restoring datasets with hash-based referencing
