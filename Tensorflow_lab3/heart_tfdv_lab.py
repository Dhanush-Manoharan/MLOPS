import os
import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import slicing_util

# ---------------- CONFIG ---------------- #

DATA_PATH = "data/heart.csv"          # main dataset
OUTPUT_DIR = "output"                 # folder for schema, etc.
SCHEMA_PATH = os.path.join(OUTPUT_DIR, "schema.pbtxt")
SLICE_SAMPLE_PATH = "slice_sample.csv"

LABEL_COL = "num"                     # heart dataset label
ID_COLS = {"id", "dataset"}           # ID-like columns
SLICE_FEATURE = "sex"                 # feature used for slicing


# ------------- DATA PREPARATION ------------- #

def prepare_data_splits_from_dataframe(df: pd.DataFrame, label_col: str):
    """
    Split full dataframe into train, eval, serving dataframes.

    - train_df: used for schema inference and model training
    - eval_df: used for evaluation statistics
    - serving_df: used to simulate serving-time data (no label column)
    """
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(0.6 * n)
    eval_end = int(0.8 * n)

    train_df = df.iloc[:train_end].copy()
    eval_df = df.iloc[train_end:eval_end].copy()
    serving_df = df.iloc[eval_end:].copy()

    # Serving data should NOT contain label column
    if label_col in serving_df.columns:
        serving_df = serving_df.drop([label_col], axis=1)

    return train_df, eval_df, serving_df


def create_slice_sample_from_dataframe(
    df: pd.DataFrame,
    slice_feature: str,
    out_path: str,
    n_per_group: int = 150,
):
    """
    Create a smaller slice sample CSV based on the main dataset.

    - Samples up to n_per_group rows per unique value of slice_feature.
    - Saves the result to slice_sample.csv.
    """
    if slice_feature not in df.columns:
        raise ValueError(
            f"Slice feature '{slice_feature}' not found in columns: {df.columns.tolist()}"
        )

    def _sample_group(g):
        return g.sample(min(len(g), n_per_group), random_state=42)

    slice_df = df.groupby(slice_feature, group_keys=False).apply(_sample_group)
    slice_df.to_csv(out_path, index=False)
    print(f"[INFO] Slice sample saved to: {out_path}")
    print(f"[INFO] Slice sample shape: {slice_df.shape}")
    return slice_df


# ------------- TFDV HELPERS ------------- #

def generate_stats_for_dataframe(df: pd.DataFrame, features_to_remove=None):
    """
    Generate TFDV statistics for a Pandas DataFrame,
    optionally dropping columns (ID_COLS).
    """
    features_to_remove = features_to_remove or set()
    approved_cols = [c for c in df.columns if c not in features_to_remove]
    df_approved = df[approved_cols].copy()
    stats = tfdv.generate_statistics_from_dataframe(df_approved)
    return stats, approved_cols


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------- Load main dataset --------- #
    print(f"[INFO] Loading main dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Full dataset shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset!")

    # --------- Create slice_sample.csv from main dataset --------- #
    _ = create_slice_sample_from_dataframe(
        df=df,
        slice_feature=SLICE_FEATURE,
        out_path=SLICE_SAMPLE_PATH,
        n_per_group=150,
    )

    # --------- Prepare train / eval / serving splits --------- #
    train_df, eval_df, serving_df = prepare_data_splits_from_dataframe(df, LABEL_COL)
    print(f"[INFO] Train shape:   {train_df.shape}")
    print(f"[INFO] Eval shape:    {eval_df.shape}")
    print(f"[INFO] Serving shape: {serving_df.shape}")

    # --------- Generate statistics --------- #
    # Training stats (used to infer schema)
    train_stats, approved_cols = generate_stats_for_dataframe(
        train_df, features_to_remove=ID_COLS
    )
    print("[INFO] Generated training statistics.")

    # Evaluation stats
    eval_stats, _ = generate_stats_for_dataframe(
        eval_df, features_to_remove=ID_COLS
    )
    print("[INFO] Generated evaluation statistics.")

    # Serving stats (no label)
    serving_stats, _ = generate_stats_for_dataframe(
        serving_df, features_to_remove=ID_COLS
    )
    print("[INFO] Generated serving statistics.")

    # --------- Infer and write schema --------- #
    schema = tfdv.infer_schema(train_stats)
    print("[INFO] Inferred schema from training statistics.")

    # Mark environments
    schema.default_environment.append("TRAINING")
    schema.default_environment.append("SERVING")

    # Label should not be present in SERVING environment
    tfdv.get_feature(schema, LABEL_COL).not_in_environment.append("SERVING")

    # Write schema.pbtxt (will overwrite existing one)
    tfdv.write_schema_text(schema, SCHEMA_PATH)
    print(f"[INFO] Schema written to: {SCHEMA_PATH}")

    # --------- Validate statistics for different environments --------- #
    print("[INFO] Validating training stats (TRAINING environment)...")
    train_anomalies = tfdv.validate_statistics(
        statistics=train_stats, schema=schema, environment="TRAINING"
    )
    tfdv.display_anomalies(train_anomalies)

    print("[INFO] Validating serving stats (SERVING environment)...")
    serving_anomalies = tfdv.validate_statistics(
        statistics=serving_stats, schema=schema, environment="SERVING"
    )
    tfdv.display_anomalies(serving_anomalies)

    # --------- Slice-based statistics --------- #
    print(f"[INFO] Generating slice-based statistics for feature: {SLICE_FEATURE}")

    slice_fn = slicing_util.get_feature_value_slicer(
        features={SLICE_FEATURE: None}
    )

    sliced_stats = tfdv.generate_statistics_from_dataframe(
        dataframe=train_df[[c for c in approved_cols if c in train_df.columns]],
        stats_options=tfdv.StatsOptions(
            schema=schema,
            slice_functions=[slice_fn],
        ),
    )

    print("[INFO] Slices present in statistics:")
    for s in sliced_stats.datasets:
        print("  -", s.name)

    print("[INFO] DONE. slice_sample.csv and output/schema.pbtxt are ready for the lab.")


if __name__ == "__main__":
    main()
