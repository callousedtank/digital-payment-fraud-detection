# Digital Payment Fraud Detection

An end-to-end machine learning system for detecting fraudulent digital payment transactions — from preprocessing and imbalance-aware training to a versioned, observable, containerized inference API and optional Streamlit client.

The project goes beyond a notebook workflow. It includes reproducible preprocessing, model versioning and rollback, experiment tracking, API validation, structured logging, automated tests, CI validation, and containerization.

## Highlights

* **Imbalance-aware training** — SMOTENC handles categorical and numerical features while preserving the correct train/test separation.
* **Deterministic preprocessing** — categorical encoders are fit only on training data and persisted with the model artifact, with unknown-category handling at inference time.
* **Model versioning and rollback** — training runs produce versioned artifacts and maintain an active model registry so evaluated versions can be re-activated without retraining.
* **Experiment tracking** — training runs record dataset fingerprints, model configuration, feature information, and validation metrics in JSONL format.
* **Production-shaped API** — FastAPI with Pydantic validation, structured request logging, latency tracking, health/readiness endpoints, metrics, and version metadata.
* **Tested and CI-validated** — automated tests cover prediction, validation, unknown categories, model registry behavior, and operational endpoints. GitHub Actions validates the test suite and container build.
* **Containerized services** — Docker Compose starts both the FastAPI inference API and the Streamlit client.
* **Web interface** — the separate Streamlit client sends requests to the same `/predict` inference endpoint.

## Features

* Data preprocessing and categorical feature encoding
* Stratified train/test split
* SMOTENC for class imbalance
* Random Forest classifier
* Joblib model artifacts
* Versioned model registry with rollback support
* JSONL experiment tracking
* FastAPI `/predict` inference endpoint
* Pydantic request validation
* Unknown-category handling
* Structured request and prediction logging
* Request latency metrics
* `/health`, `/ready`, and `/metrics` endpoints
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
│   └── *.csv                    # Dataset (not tracked by Git)
├── experiments/
│   └── *.jsonl                  # Experiment records (not tracked by Git)
├── models/
│   ├── *.joblib                 # Versioned model artifacts
│   └── model_registry.json      # Active model registry
├── notebooks/
│   └── fraud_detection.ipynb    # Exploratory notebook
├── src/
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
├── render.yaml                # Render API and Streamlit service definitions
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

### 4. Dataset Setup

The dataset is intentionally excluded from Git.

Place it at:

```text
data/Digital_Payment_Fraud_Detection_Dataset.csv
```

The expected dataset contains **7,500 transactions and 15 columns**, including the `fraud_label` target.

See [`data/README.md`](data/README.md) for dataset information.

### 5. Train the Model

```bash
python -m src.train
```

Training performs schema checks, stratified splitting, categorical encoding, SMOTENC-based balancing, and model training. It reports accuracy, precision, recall, F1, ROC-AUC, and average precision; do not promote a model based on accuracy alone.

### Model-quality note

The bundled dataset is highly imbalanced and the available non-identifier features currently show very limited fraud signal. Treat the bundled demo model as a technical demonstration, not a production fraud-decision system. Before production use, retrain and evaluate with representative labelled transactions and compare PR-AUC, fraud precision/recall, F1, and the confusion matrix against the legitimate-only baseline.

Each run produces a versioned artifact and updates the local model registry:

```text
models/model-<version>.joblib
models/model_registry.json
```

An explicit version can be supplied:

```bash
python -m src.train --model-version 1.0.0
```

An evaluated version can be activated without retraining:

```bash
python -m src.train --activate-version 1.0.0
```

The API resolves the active model through the registry. A specific evaluated version can also be selected through `MODEL_VERSION`.

### Experiment Tracking

Training records the dataset SHA-256 fingerprint, model configuration, feature set, and validation metrics in:

```text
experiments/fraud-detection.jsonl
```

Compare experiment runs with:

```bash
python -m src.compare_experiments experiments/fraud-detection.jsonl
```

The current JSONL approach provides a lightweight, dependency-free experiment record. MLflow can be evaluated later if the project requires shared experiment infrastructure or a dedicated tracking UI.

### 6. Run the API

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

Operational endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/metrics
```

The root path returns API metadata. Run the separate Streamlit client for the browser-based transaction interface:

```bash
streamlit run frontend/streamlit_app.py
```

Set `API_URL` to point the client at a different API endpoint; it defaults to the public demo endpoint.

`/predict` returns the binary `fraud_prediction`, `fraud_probability`, and the model's `decision_threshold`, so clients can present both the decision and supporting risk score.

### 7. Run Tests

```bash
pytest -v
```

The test suite covers:

* Prediction requests
* Input validation
* Unknown categorical values
* Model registry resolution
* Model rollback behavior
* Health and readiness endpoints
* Metrics endpoint
* API behavior without requiring the production model artifact during CI

### 8. Containerized Setup

The project can be built and run with Docker or Podman.

#### Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

#### Podman

```bash
podman build -t fraud-api .
podman run -p 8000:8000 fraud-api
```

#### Docker Compose (API + Streamlit client)

```bash
docker compose up --build
```

The API is available at `http://127.0.0.1:8000` and the Streamlit client at `http://127.0.0.1:8501`.

The container is configured to use the runtime-provided `PORT` when deployed to platforms such as Render.

## Deployment

The supplied `render.yaml` defines two simple Render web services: the API and the Streamlit UI. Confirm that the API service URL matches the `API_URL` value before applying it. The API exposes interactive documentation at `/docs`, prediction at `/predict`, and health, readiness, and metrics endpoints at `/health`, `/ready`, and `/metrics`.

## Environment Notes

* Python **3.14** is used for development.
* Production and development dependencies are pinned separately.
* The dataset and locally trained artifacts are excluded from version control. A tracked demo artifact supports the API demonstration.
* The application can run locally with Uvicorn or inside a container.
* The tracked demo model artifact is intended for local and containerized demonstrations.

## Future Enhancements

The current project covers the core engineering and deployment pipeline. Potential next steps include:

* **Hosted release automation** — automated versioned deployments and rollback at the deployment layer.
* **Monitoring infrastructure** — export application and prediction metrics to a dedicated monitoring backend with dashboards and alerts.
* **API evolution** — formal schema migrations and compatibility guarantees as the API changes.
* **Experiment infrastructure** — evaluate MLflow if shared tracking, artifact storage, or a team-wide experiment UI becomes necessary.
* **Model lifecycle improvements** — introduce stronger promotion gates and automated validation before a model becomes the active production version.
* **Web UI improvements** — expand the deployed transaction interface with better result explanations, validation feedback, and a more polished user experience.
