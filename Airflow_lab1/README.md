# Airflow Lab – ETL + KMeans (UCI Bank Marketing)

****This lab runs an end-to-end ETL + ML pipeline in Apache Airflow using Docker Compose.
****We cluster the UCI Bank Marketing dataset with K-Means, choose the number of clusters via the elbow method (kneed), and save the trained model to a mounted working_data directory.

****DAG name: bank_marketing_etl_kmeans

### Web UI: http://localhost:8081

### Login: airflow2 / airflow2
### Contents
```python
Airflow_lab1/
├─ dags/
│  ├─ airflow.py          # Airflow DAG definition
│  └─ src/
│     └─ lab.py           # load_data, data_preprocessing, build_save_model, load_model_elbow
├─ data/
│  └─ bank-full.csv       # UCI Bank Marketing dataset (semicolon separated)
├─ working_data/          # model & artifacts (git-ignored)
├─ logs/                  # airflow logs (git-ignored)
├─ docker-compose.yaml    # stack (webserver on :8081, pg, redis, worker, scheduler, triggerer)
├─ requirements.txt       # pandas, scikit-learn, kneed, etc.
└─ README.md

```

# What the DAG does

****load_data
   Loads data/bank-full.csv (semicolon delimited) and returns a pickled pandas DataFrame via XCom.

****Data_preprocessing
   Selects numeric columns, drops NAs, applies MinMaxScaler, and returns the scaled matrix (pickled) via XCom.

****build_save_model
   Trains K-Means for a range of k to compute SSE (inertia) values, then saves the last trained model to
   /opt/airflow/working_data/model_bank.sav (host: working_data/model_bank.sav).
   Returns the SSE list via XCom.

****load_model_elbow
   Reloads the saved model, recomputes preprocessing for a quick sample prediction, and uses KneeLocator to infer the elbow k.
   Returns a string like: "Cluster 3 Number of clusters 8"

### Airflow Setup

Used Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes the tasks on an array of workers while following the specified dependencies.

References

-   Product - https://airflow.apache.org/
-   Documentation - https://airflow.apache.org/docs/
-   Github - https://github.com/apache/airflow

#### Installation

Prerequisites: You should allocate at least 4GB memory for the Docker Engine (ideally 8GB).

Local

-   Docker Desktop Running

Cloud

-   Linux VM
-   SSH Connection
-   Installed Docker Engine - [Install using the convenience script](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)

# Quickstart (Docker)
```bash
cd C:\MLOPS\Airflow_lab1
```
****First run (or after a full reset):
```bash
docker compose up airflow-init
docker compose up -d

```
****Next time (normal start):
```bash
docker compose up -d
```
### Open http://localhost:8081
    log in with airflow2 / airflow2.
      Toggle the DAG on, then click Trigger DAG (default config works).
      If you want to pass an explicit path, use:
```bash
{"source_path": "/opt/airflow/data/bank-full.csv"}
```
# Outputs
   working_data/model_bank.sav – saved KMeans model
   Final task log contains the cluster prediction for one sample and the elbow k.
   
# Stopping / Restarting
Stop and remove containers (keep data/volumes):
```bash
docker compose down
```
# Full reset (wipes DB & logs/artifacts – use with care):
```bash
docker compose down -v
rmdir /S /Q logs working_data
mkdir logs working_data
docker compose up airflow-init
docker compose up -d
```
# Check status / logs:
```bash
docker compose ps
docker compose logs -f webserver
```
# Implementation details

****All pipeline functions are in dags/src/lab.py and are written to work with Airflow’s XComs:

****load_data(source_path="/opt/airflow/data/bank-full.csv") -> bytes
****Reads the dataset with sep=';' and returns a pickled DataFrame.
****data_preprocessing(data: bytes) -> bytes
****Numeric-only + dropna + MinMaxScaler → pickled ndarray.

****build_save_model(data: bytes, filename="/opt/airflow/working_data/model_bank.sav", k_min=2, k_max=15) -> list
   Computes SSE for k in [k_min..k_max], ensures output directory exists, and saves the model (pickle).

****load_model_elbow(filename: str, sse: list, source_path="/opt/airflow/data/bank-full.csv") -> str
   Reloads model, predicts for one sample, detects elbow with KneeLocator, returns a friendly summary string.

# Tech

# Apache Airflow (webserver, scheduler, worker, triggerer)

# PostgreSQL (metadata DB) & Redis (Celery broker)

# Python: pandas, scikit-learn, kneed, joblib/pickle

# Docs:

# Airflow: https://airflow.apache.org/

# Airflow Docker Compose guide: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
