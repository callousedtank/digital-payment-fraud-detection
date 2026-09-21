# Digital Payment Fraud Detection

An end-to-end machine learning system for detecting fraudulent digital payment transactions, covering dataset evaluation, imbalance-aware training, experiment tracking, model versioning, and a validated containerized inference API.

The project evolved from an initial digital-payment fraud detection prototype into a more rigorous ML engineering workflow. The original dataset was evaluated and found to contain very weak predictive signal, so a public credit-card fraud benchmark was added to separate dataset limitations from model and pipeline behavior.

## Live Demo

* **API:** https://digital-payment-fraud-detection.onrender.com
* **Interactive API docs:** https://digital-payment-fraud-detection.onrender.com/docs
* **Frontend:** Streamlit client runs locally; public deployment is pending.

## Highlights

* **Multiple dataset support** — the training pipeline supports both the original digital-payment dataset and the ULB/Worldline credit-card fraud benchmark through a shared dataset configuration.
* **Dataset-specific preprocessing** — target columns, identifier columns, and resampling methods are configured per dataset rather than hardcoded into the training pipeline.
* **Imbalance-aware training** — supports `SMOTENC` for the original dataset with categorical features and `SMOTE` for the numerical ULB benchmark, while keeping resampling restricted to the training split.
* **Deterministic preprocessing** — categorical encoders are fit only on training data and persisted with the model artifact, with unknown-category handling at inference time.
* **Model versioning and rollback** — training runs produce versioned artifacts and maintain a model registry so evaluated versions can be re-activated without retraining.
* **Experiment tracking** — training runs record dataset fingerprints, model configuration, feature information, and validation metrics in JSONL format.
* **Production-shaped API** — FastAPI with Pydantic validation, structured request logging, latency tracking, health/readiness endpoints, metrics, and model/version metadata.
* **Tested and CI-validated** — automated tests cover prediction, validation, model registry behavior, experiment tracking, and operational endpoints.
* **Containerized services** — the inference API can be built and run with Docker or Podman.
* **Web interface** — the separate Streamlit client sends requests to the same `/predict` inference endpoint.

## Features

* Multiple dataset configurations
* Dataset schema and target validation
* Stratified train/test splitting
* Dataset-specific categorical encoding
* SMOTENC and SMOTE resampling
* Random Forest and Logistic Regression training
* Joblib model artifacts
* Versioned model registry with rollback support
* JSONL experiment tracking
* Dataset SHA-256 fingerprinting
* FastAPI `/predict` inference endpoint
* Pydantic request validation
* Unknown-category handling
* Structured request and prediction logging
* Request latency metrics
* `/health`, `/ready`, `/model`, and `/metrics` endpoints
* Schema and model version metadata
* Automated API and infrastructure tests with Pytest
* GitHub Actions CI
* Containerized deployment
* Docker Compose API and Streamlit client setup

## Project Structure

```text
digital-payment-fraud-detection/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   └── main.py                  # FastAPI application
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── streamlit_app.py         # Streamlit client
├── data/
│   ├── README.md                # Dataset information
│   └── *.csv                    # Datasets (not tracked by Git)
├── experiments/
│   └── *.jsonl                  # Experiment records (not tracked by Git)
├── models/
│   ├── *.joblib                 # Locally trained model artifacts
│   └── model_registry.json      # Model registry
├── notebooks/
│   └── fraud_detection.ipynb    # Exploratory notebook
├── src/
│   ├── datasets.py              # Dataset configuration and validation
│   ├── preprocessing.py         # Data preprocessing
│   ├── train.py                 # Training and model versioning
│   ├── predict.py               # Inference
│   ├── model_registry.py        # Model resolution and rollback
│   ├── experiment_tracking.py   # Experiment recording
│   └── compare_experiments.py   # Experiment comparison
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_model_registry.py
│   └── test_experiment_tracking.py
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/callousedtank/digital-payment-fraud-detection.git
cd digital-payment-fraud-detection
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows — Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### Windows — PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## Dataset Setup

The datasets are intentionally excluded from Git.

The project currently supports two datasets.

### Original Digital-Payment Dataset

```text
data/Digital_Payment_Fraud_Detection_Dataset.csv
```

The original dataset contains:

* 7,500 transactions
* 15 columns
* `fraud_label` as the target
* 6.52% fraud rate

The identifier columns `transaction_id` and `user_id` are excluded during preprocessing.

### ULB Credit-Card Fraud Benchmark

```text
data/creditcard.csv
```

The ULB/Worldline dataset contains:

* 284,807 transactions
* 31 columns
* `Class` as the target
* 492 fraud transactions
* approximately 0.173% fraud rate

The ULB dataset contains numerical features and therefore uses `SMOTE` when resampling is enabled.

See [`data/README.md`](data/README.md) for dataset information.

## Dataset Configuration

Dataset-specific configuration is defined in:

```text
src/datasets.py
```

The training pipeline accepts:

```text
original
ulb
```

For example:

```bash
python -m src.train --dataset original
```

or:

```bash
python -m src.train --dataset ulb
```

The dataset configuration determines the dataset path, target column, and identifier columns.

## Training

### Original Dataset

Random Forest with SMOTENC:

```bash
python -m src.train \
    --dataset original \
    --model-type random_forest \
    --resampling smotenc
```

### ULB Benchmark Without Resampling

```bash
python -m src.train \
    --dataset ulb \
    --model-type random_forest \
    --resampling none
```

### ULB Benchmark With SMOTE

```bash
python -m src.train \
    --dataset ulb \
    --model-type random_forest \
    --resampling smote
```

Training performs schema validation, stratified splitting, dataset-specific encoding, and optional training-set-only resampling.

The evaluation reports:

* Accuracy
* Precision
* Recall
* F1
* PR-AUC
* ROC-AUC
* Confusion matrix

For fraud detection, accuracy is not treated as the primary model-quality measure because the positive class is highly imbalanced.

## Dataset Benchmark Results

The original dataset was evaluated first and showed very weak predictive signal.

### Original Dataset — Random Forest + SMOTENC

| Metric          | Result |
| --------------- | -----: |
| Accuracy        |  0.844 |
| Fraud Precision |  0.085 |
| Fraud Recall    |  0.143 |
| Fraud F1        |  0.107 |
| PR-AUC          |  0.064 |
| ROC-AUC         |  0.477 |

The result was not interpreted as a model-specific failure. Additional analysis of the original dataset showed very weak relationships between the available features and the fraud target.

A second dataset was therefore introduced to provide a stronger fraud-detection benchmark.

### ULB Benchmark — Random Forest

| Configuration | ROC-AUC | PR-AUC | Fraud Precision | Fraud Recall | Fraud F1 |
| ------------- | ------: | -----: | --------------: | -----------: | -------: |
| No resampling |   0.963 |  0.873 |           0.941 |        0.816 |    0.874 |
| SMOTE         |   0.964 |  0.875 |           0.835 |        0.827 |    0.831 |

The ULB results show that the same inference and evaluation pipeline produces substantially stronger results when evaluated on a dataset with useful predictive signal.

SMOTE produced a small improvement in ROC-AUC, PR-AUC, and recall, but reduced precision and F1 at the default `0.5` decision threshold. Therefore, resampling is treated as an experimental configuration rather than an automatic improvement.

### Interpretation

The benchmark separates two different concerns:

```text
Model / pipeline behavior
        +
Dataset quality
        ↓
Observed evaluation result
```

A weak result on the original dataset should not automatically be interpreted as evidence that the entire ML pipeline is ineffective.

The ULB benchmark provides a second reference point with a substantially different dataset and feature space.

The ULB dataset is a public benchmark and should not be interpreted as representative of a real digital-payment production environment.

## Model Versioning

Each training run produces a versioned model artifact and records it in:

```text
models/model-<version>.joblib
models/model_registry.json
```

Model artifacts are excluded from normal Git tracking.

The registry records model metadata such as:

* model version
* artifact path
* classifier
* dataset
* training configuration
* validation metrics
* validation status

An explicit version can be supplied during training:

```bash
python -m src.train --model-version 1.0.0
```

An evaluated version can be activated without retraining:

```bash
python -m src.train --activate-version 1.0.0
```

The API resolves the active model through the registry.

A specific model version can also be selected explicitly through the `MODEL_VERSION` environment variable:

```bash
MODEL_VERSION=20260921T143704Z python -m src.predict
```

This allows the registry's default active version and an explicitly selected evaluated version to be used independently.

## Experiment Tracking

Training records:

* dataset SHA-256 fingerprint
* dataset name
* model configuration
* resampling configuration
* feature set
* validation metrics
* training configuration

Experiment records are stored in:

```text
experiments/fraud-detection.jsonl
```

Compare experiment runs with:

```bash
python -m src.compare_experiments experiments/fraud-detection.jsonl
```

The current JSONL approach provides a lightweight experiment-tracking format without requiring a dedicated tracking platform. MLflow can be evaluated later if shared experiment infrastructure or a dedicated tracking UI becomes necessary.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The local API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Operational Endpoints

```text
/health
/ready
/model
/metrics
```

### Prediction Endpoint

```text
/predict
```

The API validates incoming requests before inference.

Invalid requests can be rejected when they contain:

* missing required features
* unexpected features
* non-finite numeric values

`/predict` returns:

* `fraud_prediction`
* `fraud_probability`
* `decision_threshold`
* `model_version`
* classifier information
* API schema version

The `/model` endpoint exposes the currently resolved model metadata and feature schema.

Run the separate Streamlit client for the browser-based transaction interface:

```bash
streamlit run frontend/streamlit_app.py
```

Set `API_URL` to point the client at a different API endpoint.

## Testing

Run the test suite with:

```bash
pytest -v
```

The current test suite covers:

* Prediction requests
* Input validation
* Unknown categorical values
* Dataset preprocessing
* Model registry resolution
* Model version selection
* Model rollback behavior
* Health and readiness endpoints
* Metrics endpoint
* API behavior without requiring the production model artifact during CI

Current local verification:

```text
16 passed
```

## Containerized Setup

The project can be built and run with Docker or Podman.

### Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

### Podman

```bash
podman build -t fraud-api .
podman run -p 8000:8000 fraud-api
```

The containerized API was verified locally using Podman with:

```text
/health
/ready
/model
/predict
```

The API also correctly rejects invalid prediction requests with HTTP 400 responses.

### Docker Compose

```bash
docker compose up --build
```

The API is available at:

```text
http://127.0.0.1:8000
```

and the Streamlit client at:

```text
http://127.0.0.1:8501
```

The container is configured to use the runtime-provided `PORT` when deployed to platforms such as Render.

## Deployment

The supplied `render.yaml` defines the deployment configuration for the API and Streamlit client.

The API exposes:

```text
/docs
/predict
/health
/ready
/model
/metrics
```

The deployed API should be treated as a demonstration environment rather than a production fraud-decision service.

## Environment Notes

* Python **3.14** is used for development.
* Production and development dependencies are maintained separately.
* Datasets and locally trained model artifacts are excluded from version control.
* The model registry is tracked, while locally generated `.joblib` artifacts are ignored.
* The application can run locally with Uvicorn or inside a container.
* The tracked demo model artifact is intended for local and containerized demonstrations.
* The ULB benchmark results are evaluation results on a public dataset and are not evidence of production performance on real digital-payment traffic.

## Limitations

The project is an engineering and benchmarking system rather than a production fraud-detection service.

Important limitations include:

* The original digital-payment dataset contains weak predictive signal.
* The ULB dataset is a public credit-card fraud benchmark and differs from real digital-payment traffic.
* No real financial institution transaction stream is used.
* No production customer data is used.
* The deployed system should not be used to make real financial decisions.
* Threshold selection has not been optimized for a specific operational fraud-loss/cost tradeoff.
* Model monitoring and automated promotion gates are not yet implemented as a complete production monitoring system.

## Future Enhancements

Potential next steps include:

* **Public frontend deployment** — deploy the Streamlit client and connect it to the live API.
* **Threshold optimization** — evaluate decision thresholds against explicit precision/recall and operational cost requirements.
* **Model lifecycle improvements** — introduce stronger promotion gates and automated validation before a model becomes active.
* **Monitoring infrastructure** — export application and prediction metrics to a dedicated monitoring backend with dashboards and alerts.
* **Hosted release automation** — automate versioned model deployment and rollback at the deployment layer.
* **API evolution** — formal schema migrations and compatibility guarantees as the API changes.
* **Experiment infrastructure** — evaluate MLflow if shared tracking, artifact storage, or a team-wide experiment UI becomes necessary.
* **Web UI improvements** — expand the transaction interface with better result explanations, validation feedback, and model information.
