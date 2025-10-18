import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator


# ---------- 1) Load data ----------

def load_data(source_path: str = "/opt/airflow/data/bank-full.csv") -> bytes:
    """
    Load the raw dataset and return it serialized (bytes) for XCom.
    Bank Marketing data uses ';' as the delimiter.

    Returns:
        bytes: pickled pandas DataFrame
    """
    df = pd.read_csv(source_path, sep=";")
    return pickle.dumps(df)


# ---------- 2) Preprocess ----------

def data_preprocessing(data: bytes) -> bytes:
    """
    Deserialize the DataFrame, select numeric features, drop NAs,
    MinMax-scale, and return the scaled array serialized.

    Args:
        data: pickled pandas DataFrame

    Returns:
        bytes: pickled numpy array (scaled features)
    """
    df: pd.DataFrame = pickle.loads(data)

    # numeric-only features; simple NA handling
    X = df.select_dtypes(include=["number"]).dropna()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return pickle.dumps(X_scaled)


# ---------- 3) Train & save model ----------

def build_save_model(data: bytes,
                     filename: str = "/opt/airflow/working_data/model_bank.sav",
                     k_min: int = 2,
                     k_max: int = 15) -> list:
    """
    Fit KMeans for a range of k to compute SSE, and save the last trained model.
    Ensures the output directory exists before writing.

    Args:
        data: pickled numpy array (scaled features)
        filename: where to save the model
        k_min, k_max: inclusive k range for SSE curve

    Returns:
        list: SSE values for k in [k_min, k_max]
    """
    X = pickle.loads(data)

    # KMeans kwargs chosen for stability
    km_args = dict(init="random", n_init=10, max_iter=300, random_state=42)
    sse = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, **km_args)
        km.fit(X)
        sse.append(km.inertia_)

    # ensure output dir is writable and exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(km, f)

    return sse


# ---------- 4) Load model & elbow ----------

def load_model_elbow(filename: str,
                     sse: list,
                     source_path: str = "/opt/airflow/data/bank-full.csv") -> str:
    """
    Load the saved model, recompute preprocessing on the dataset,
    predict a cluster for the first row, and compute the elbow k.

    Args:
        filename: path to saved KMeans
        sse: SSE list (output from build_save_model)
        source_path: dataset path (used to build a sample to predict)

    Returns:
        str: "Cluster X Number of clusters Y"
    """
    # load model
    with open(filename, "rb") as f:
        model: KMeans = pickle.load(f)

    # rebuild the same preprocessing for a prediction sample
    df = pd.read_csv(source_path, sep=";")
    X = df.select_dtypes(include=["number"]).dropna()
    X_scaled = MinMaxScaler().fit_transform(X)

    # take the first row as a small demo prediction
    sample = X_scaled[:1]
    pred = model.predict(sample)[0]

    # elbow from SSE
    k_range = list(range(2, 2 + len(sse)))  # matches k_min..k_max used above
    kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
    elbow_k = kl.elbow if kl.elbow is not None else k_range[-1]

    return f"Cluster {pred} Number of clusters {elbow_k}"
