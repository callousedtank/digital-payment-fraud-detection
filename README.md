# Digital Payment Fraud Detection

A machine learning system for detecting fraudulent digital payment transactions.

The project takes a trained machine learning model and exposes it through a FastAPI REST API, with automated tests and containerized deployment.

## Features

- Data preprocessing and categorical feature encoding
- Stratified train/test split
- SMOTENC for handling class imbalance
- Random Forest classifier
- Model artifact generation with Joblib
- FastAPI `/predict` endpoint
- Request validation with Pydantic
- Structured request/prediction logging and latency metrics
- Health (`/health`), readiness (`/ready`), and metrics (`/metrics`) endpoints
- Versioned model artifacts, metadata, and rollback support
- Reproducible JSONL experiment tracking and model comparison
- Automated API tests with Pytest
- GitHub Actions test and container-build validation
- Containerized deployment with Podman

## Project Structure

```text
digital-payment-fraud-detection/
├── app/
│   └── main.py                 # FastAPI application
├── data/
│   ├── README.md               # Dataset information
│   └── *.csv                   # Dataset (not tracked by Git)
├── models/
│   └── *.joblib                # Trained model (not tracked by Git)
├── notebooks/
│   └── fraud_detection.ipynb   # Original exploratory notebook
├── src/
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── train.py                # Model training
│   └── predict.py              # Model prediction
├── tests/
│   └── test_api.py             # API tests
├── Dockerfile
├── docker-compose.yml
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

A virtual environment is recommended to keep project dependencies isolated from the system Python installation.

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

> **Note:** Python executable names differ between operating systems. Linux and macOS commonly use `python3`, while Windows commonly uses `python`.

### 3. Install Python Dependencies

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements.txt file contains the pinned production dependencies used by the project.

For development and testing, install the additional development dependencies:

```bash
pip install -r requirements-dev.txt
```

The requirements-dev.txt file includes the production dependencies plus development and testing tools such as Pytest and HTTPX.

### 4. Dataset Setup

The dataset is intentionally **not included in the Git repository**.

Place the dataset at:

```text
data/Digital_Payment_Fraud_Detection_Dataset.csv
```

The expected dataset contains **7,500 transactions and 15 columns**, including the `fraud_label` target column.

See [`data/README.md`](data/README.md) for dataset information.

### 5. Generate the Model

The trained model is also intentionally excluded from Git. Generate it locally by running:

```bash
python src/train.py
```

This performs preprocessing, train/test splitting, categorical encoding, SMOTENC-based class balancing, and Random Forest training.

Each training run writes an immutable, versioned model artifact and updates the
local model registry. Without an explicit version, the version is a UTC timestamp:

```text
models/model-<version>.joblib
models/model_registry.json
```

Use an explicit version when training a release candidate:

```bash
python src/train.py --model-version 1.0.0
```

To roll back the active version after it has been validated:

```bash
python src/train.py --activate-version 1.0.0
```

The API reads the active model from the registry. Set `MODEL_VERSION` to serve a
specific validated version without changing the registry. Existing
`models/model.joblib` artifacts remain supported as a legacy fallback.

### Experiment Tracking

Training records the dataset SHA-256 fingerprint, model configuration, feature
set, and validation metrics in `experiments/fraud-detection.jsonl`. These local
records are intentionally excluded from Git because they may identify private
datasets.

Train and compare a second model family with:

```bash
python src/train.py --model-type logistic_regression --model-version logistic-1
python src/compare_experiments.py experiments/fraud-detection.jsonl
```

The JSONL record format provides a dependency-free, reproducible baseline. MLflow
is the next tool to evaluate when a shared experiment server, artifact store, or
team-wide experiment UI is needed.

### 6. Run the API

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Operational endpoints are available at:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/metrics
```

### 7. Run Tests

With the virtual environment activated:

```bash
pytest -v
```

The test suite verifies API prediction requests and validation of missing input fields.

### 8. Containerized Setup

The project can also be built and run as a container.

The repository uses a standard `Dockerfile` and is compatible with container engines such as **Docker** and **Podman**.

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

> **OS note:** Docker and Podman installation is platform-specific. The commands used to build and run this project remain largely the same once the container engine is installed.

## Environment Notes

* Python **3.14** is currently used for development.
* Dependencies are pinned in `requirements.txt` and `requirements-dev.txt` for reproducibility.
* The project uses a virtual environment and does not require modifying the system Python installation.
* Dataset and trained model artifacts are excluded from version control through `.gitignore`.
* The API can be run directly with Uvicorn or inside a container.

## Future Enhancements

The current project provides a working local and containerized ML inference pipeline. Future development will focus on improving its production-readiness and ML engineering capabilities.

### 1. CI/CD

- GitHub Actions runs the test suite on every push and pull request
- GitHub Actions validates that the container image builds successfully
- Add deployment and release automation

### 2. Logging & Monitoring

- Structured API logging, request/prediction latency, API-error counters, and health/readiness endpoints are available
- Expose metrics to a monitoring backend and add dashboards/alerts

### 3. Model & Schema Versioning

- Versioned artifact filenames, metadata, and rollback registry are available
- API responses and readiness checks expose schema version `1.0.0`
- Add schema migrations and compatibility guarantees for later API versions

### 4. Experiment Tracking

- JSONL records capture model configuration, dataset fingerprint, and evaluation metrics
- Random Forest and Logistic Regression runs can be compared with the comparison CLI
- Evaluate MLflow when shared tracking infrastructure is required

### Planned Progression

```text
Current
  │
  ├── ML Pipeline
  ├── FastAPI Inference API
  ├── Automated Tests
  └── Containerized Deployment
        │
        ▼
Future
  │
  ├── CI/CD
  ├── Logging & Monitoring
  ├── Model & Schema Versioning
  └── Experiment Tracking
