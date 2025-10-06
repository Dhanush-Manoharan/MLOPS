FROM apache/airflow:2.5.1

# Install as non-root 'airflow' user
USER airflow

ARG AIRFLOW_VERSION=2.5.1
ARG PYTHON_VERSION=3.7
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

RUN python --version

# Py3.7-compatible pins
RUN python -m pip install --no-cache-dir pandas==1.3.5 scikit-learn==1.0.2 kneed --constraint "${CONSTRAINT_URL}"

USER airflow
