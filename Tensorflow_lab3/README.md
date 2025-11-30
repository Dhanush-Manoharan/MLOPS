##  TensorFlow Data Validation (TFDV) – Heart Disease Dataset

### Course: MLOps

### Lab: TensorFlow Data Validation

### Dataset: Heart Disease Dataset


##  Overview

This lab demonstrates the use of **TensorFlow Data Validation (TFDV)** to analyze, validate, and monitor a structured tabular dataset as part of an MLOps data validation pipeline.

The workflow follows the same steps as the professor’s assignment notebook, implemented as a **reproducible Python script** suitable for real-world ML pipelines.

Key objectives:

* Generate dataset statistics
* Infer a schema from training data
* Validate serving data against the schema
* Perform slice-based data analysis


##  Project Structure

```
Tensorflow_lab3/
│
├── data/
│   └── heart.csv                # Input dataset
│
├── output/
│   └── schema.pbtxt             # Inferred TFDV schema
│
├── slice_sample.csv             # Slice-based sample dataset
├── heart_tfdv_lab.py             # Main TFDV pipeline script
├── .gitignore
└── README.md
```

##  Dataset Description

* **Rows:** 920
* **Columns:** 16
* **Label column:** `num`
* **Slice feature:** `sex`

Selected ID-like columns (`id`, `dataset`) are excluded from schema inference.



##  TFDV Pipeline Steps

###  Load Dataset

* Reads `heart.csv`
* Verifies label presence

###  Data Splitting

* 60% Training
* 20% Evaluation
* 20% Serving (label removed)

###  Statistics Generation

* Training statistics (used for schema inference)
* Evaluation statistics
* Serving statistics

###  Schema Inference

* Schema inferred from training data using TFDV
* Environments defined:

  * `TRAINING`
  * `SERVING`
* Label column excluded from SERVING

###  Data Validation

* Training data validated against schema
* Serving data validated against schema
* Anomalies displayed (CLI output)

###  Slice-Based Analysis

* Slice feature: `sex`
* Slice-based statistics generated using `slicing_util`


##  How to Run

### 1️ Activate Virtual Environment

```bash
.\.venv\Scripts\activate
```

### 2️ Run the Script

```bash
python heart_tfdv_lab.py
```


##  Outputs Generated

| File                  | Description                                     |
| --------------------- | ----------------------------------------------- |
| `output/schema.pbtxt` | Inferred TFDV schema                            |
| `slice_sample.csv`    | Sample dataset used for slice analysis          |
| Console logs          | Training, evaluation, serving validation status |

Successful execution ends with:

```
[INFO] DONE. slice_sample.csv and output/schema.pbtxt are ready for the lab.
```



##  Notes

* Visualization warnings appear if `IPython` is not installed.
* These warnings do **not** affect schema inference or validation.
* The lab executes fully in CLI mode.



##  Conclusion

This lab successfully applies **TensorFlow Data Validation** to:

* Detect schema issues
* Validate serving data safety
* Perform slice-based data analysis
