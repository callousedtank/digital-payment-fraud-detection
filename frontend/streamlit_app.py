import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

ORIGINAL_DATASET_PATH = Path(
    os.getenv(
        "ORIGINAL_DATASET_PATH",
        PROJECT_ROOT
        / "data"
        / "Digital_Payment_Fraud_Detection_Dataset.csv",
    )
)

ULB_DATASET_PATH = Path(
    os.getenv(
        "ULB_DATASET_PATH",
        PROJECT_ROOT
        / "data"
        / "creditcard.csv",
    )
)

DEFAULT_API_URL = (
    "http://127.0.0.1:8000/predict"
)

API_URL = os.getenv(
    "API_URL",
    DEFAULT_API_URL,
)


DATASET_CONFIG = {
    "Original Digital Payment": {
        "path": ORIGINAL_DATASET_PATH,
        "target": "fraud_label",
        "model_version": "20260921T143524Z",
    },
    "ULB Credit Card Benchmark": {
        "path": ULB_DATASET_PATH,
        "target": "Class",
        "model_version": "20260921T143704Z",
    },
}


ORIGINAL_FEATURES = [
    "transaction_amount",
    "transaction_type",
    "payment_mode",
    "device_type",
    "device_location",
    "account_age_days",
    "transaction_hour",
    "previous_failed_attempts",
    "avg_transaction_amount",
    "is_international",
    "ip_risk_score",
    "login_attempts_last_24h",
]


ULB_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


st.set_page_config(
    page_title="Digital Payment Fraud Detection",
    page_icon="💳",
    layout="centered",
)


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


def get_dataset_config(dataset_name):
    return DATASET_CONFIG[
        dataset_name
    ]


def select_random_transaction(
    dataframe,
    target_column,
    class_filter,
):
    if class_filter == "Any":
        candidates = dataframe

    elif class_filter == "Legitimate":
        candidates = dataframe[
            dataframe[target_column] == 0
        ]

    else:
        candidates = dataframe[
            dataframe[target_column] == 1
        ]

    if candidates.empty:
        raise ValueError(
            "No transactions match the selected class."
        )

    return candidates.sample(
        n=1
    ).iloc[0]


def get_ground_truth(
    transaction,
    target_column,
):
    return int(
        transaction[target_column]
    )


def build_ulb_payload(transaction):
    return {
        "features": {
            feature: float(
                transaction[feature]
            )
            for feature in ULB_FEATURES
        }
    }


def build_original_payload_from_form(
    transaction_amount,
    transaction_type,
    payment_mode,
    device_type,
    device_location,
    account_age_days,
    transaction_hour,
    previous_failed_attempts,
    avg_transaction_amount,
    is_international,
    ip_risk_score,
    login_attempts_last_24h,
):
    return {
        "features": {
            "transaction_amount": transaction_amount,
            "transaction_type": transaction_type,
            "payment_mode": payment_mode,
            "device_type": device_type,
            "device_location": device_location,
            "account_age_days": account_age_days,
            "transaction_hour": transaction_hour,
            "previous_failed_attempts": previous_failed_attempts,
            "avg_transaction_amount": avg_transaction_amount,
            "is_international": is_international,
            "ip_risk_score": ip_risk_score,
            "login_attempts_last_24h": login_attempts_last_24h,
        }
    }


def analyze_transaction(
    payload,
    model_version,
):
    request_payload = {
        **payload,
        "model_version": model_version,
    }

    response = requests.post(
        API_URL,
        json=request_payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


st.title("Transaction fraud review")

st.caption(
    "Evaluate transactions using the selected fraud-detection "
    "dataset and its registered model."
)


with st.expander(
    "How this demo works",
    expanded=False,
):
    st.write(
        "Choose a dataset, select how the transaction should be "
        "provided, and send it to the versioned inference API."
    )

    st.write(
        "The ULB dataset uses anonymized PCA-derived features. "
        "Those features are kept hidden from the user and are "
        "taken directly from real benchmark transactions."
    )

    st.write(
        "The original dataset uses human-readable transaction "
        "fields and supports manual transaction entry."
    )


st.subheader("1. Select dataset")

dataset_name = st.selectbox(
    "Dataset",
    list(
        DATASET_CONFIG.keys()
    ),
)

dataset_config = get_dataset_config(
    dataset_name
)

model_version = dataset_config[
    "model_version"
]

dataset_path = dataset_config[
    "path"
]

target_column = dataset_config[
    "target"
]


if not dataset_path.exists():
    st.error(
        f"Dataset not found: `{dataset_path}`"
    )
    st.stop()


try:
    dataset = load_dataset(
        dataset_path
    )
except Exception as exc:
    st.error(
        f"Failed to load dataset: {exc}"
    )
    st.stop()


st.caption(
    f"Model version: `{model_version}`"
)


st.subheader("2. Select transaction")

transaction_mode = st.radio(
    "Transaction source",
    [
        "Automatic sample",
        "Manual selection",
    ],
    horizontal=True,
)


selected_transaction = None
selected_ground_truth = None
payload = None


if transaction_mode == "Automatic sample":

    class_filter = st.selectbox(
        "Sample class",
        [
            "Any",
            "Legitimate",
            "Fraud",
        ],
    )

    if st.button(
        "Load random transaction",
        use_container_width=True,
    ):
        try:
            selected_transaction = (
                select_random_transaction(
                    dataset,
                    target_column,
                    class_filter,
                )
            )

            st.session_state[
                "selected_transaction"
            ] = selected_transaction.to_dict()

            st.session_state[
                "selected_dataset"
            ] = dataset_name

        except ValueError as exc:
            st.error(str(exc))


else:

    if dataset_name == "ULB Credit Card Benchmark":

        transaction_index = st.number_input(
            "Dataset row index",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1,
        )

        if st.button(
            "Load transaction",
            use_container_width=True,
        ):
            selected_transaction = dataset.iloc[
                int(transaction_index)
            ]

            st.session_state[
                "selected_transaction"
            ] = selected_transaction.to_dict()

            st.session_state[
                "selected_dataset"
            ] = dataset_name

    else:

        with st.form(
            "original_manual_transaction"
        ):
            st.write(
                "Enter the transaction fields used by "
                "the original fraud model."
            )

            transaction_amount = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=5000.0,
                step=100.0,
            )

            transaction_type = st.selectbox(
                "Transaction type",
                [
                    "Payment",
                    "Transfer",
                    "Withdrawal",
                ],
            )

            payment_mode = st.selectbox(
                "Payment mode",
                [
                    "Card",
                    "NetBanking",
                    "UPI",
                    "Wallet",
                ],
            )

            device_type = st.selectbox(
                "Device type",
                [
                    "Android",
                    "iOS",
                    "Web",
                ],
            )

            device_location = st.text_input(
                "Device location",
                value="Chennai",
            )

            account_age_days = st.number_input(
                "Account age (days)",
                min_value=0,
                value=365,
                step=1,
            )

            transaction_hour = st.slider(
                "Transaction hour",
                min_value=0,
                max_value=23,
                value=14,
            )

            previous_failed_attempts = (
                st.number_input(
                    "Previous failed attempts",
                    min_value=0,
                    value=0,
                    step=1,
                )
            )

            avg_transaction_amount = (
                st.number_input(
                    "Average transaction amount",
                    min_value=0.0,
                    value=4500.0,
                    step=100.0,
                )
            )

            is_international = st.selectbox(
                "International transaction?",
                options=[0, 1],
                format_func=lambda value:
                "Yes" if value else "No",
            )

            ip_risk_score = st.slider(
                "IP risk score",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
            )

            login_attempts_last_24h = (
                st.number_input(
                    "Login attempts in last 24h",
                    min_value=0,
                    value=2,
                    step=1,
                )
            )

            manual_submitted = (
                st.form_submit_button(
                    "Prepare transaction"
                )
            )

        if manual_submitted:
            payload = (
                build_original_payload_from_form(
                    transaction_amount,
                    transaction_type,
                    payment_mode,
                    device_type,
                    device_location,
                    account_age_days,
                    transaction_hour,
                    previous_failed_attempts,
                    avg_transaction_amount,
                    is_international,
                    ip_risk_score,
                    login_attempts_last_24h,
                )
            )

            st.session_state[
                "manual_payload"
            ] = payload

            st.session_state[
                "selected_dataset"
            ] = dataset_name


if (
    st.session_state.get(
        "selected_dataset"
    )
    == dataset_name
):
    if "selected_transaction" in st.session_state:
        selected_transaction = pd.Series(
            st.session_state[
                "selected_transaction"
            ]
        )

    payload = st.session_state.get(
        "manual_payload",
        payload,
    )


if selected_transaction is not None:

    selected_ground_truth = (
        get_ground_truth(
            selected_transaction,
            target_column,
        )
    )

    st.divider()

    st.subheader(
        "Selected benchmark transaction"
    )

    if dataset_name == "ULB Credit Card Benchmark":

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Amount",
                f"${float(selected_transaction['Amount']):,.2f}",
            )

        with col2:
            st.metric(
                "Time",
                f"{float(selected_transaction['Time']):,.0f}",
            )

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Transaction amount",
                f"{float(selected_transaction['transaction_amount']):,.2f}",
            )

        with col2:
            st.write(
                "**Transaction type**"
            )
            st.write(
                selected_transaction[
                    "transaction_type"
                ]
            )


if (
    transaction_mode == "Manual selection"
    and dataset_name
    == "Original Digital Payment"
    and "manual_payload" in st.session_state
):

    st.divider()

    st.subheader(
        "Prepared transaction"
    )

    st.info(
        "Manual transaction is ready for inference."
    )


    payload = st.session_state[
        "manual_payload"
    ]


if (
    payload is None
    and selected_transaction is not None
):

    if dataset_name == "ULB Credit Card Benchmark":
        payload = build_ulb_payload(
            selected_transaction
        )

    else:
        payload = {
            "features": {
                feature: selected_transaction[
                    feature
                ]
                for feature in ORIGINAL_FEATURES
            }
        }


if payload is not None:

    st.divider()

    if st.button(
        "Analyze transaction",
        type="primary",
        use_container_width=True,
    ):

        try:
            result = analyze_transaction(
                payload,
                model_version,
            )

        except requests.RequestException as exc:
            st.error(
                f"Prediction request failed: {exc}"
            )

        else:

            prediction = result.get(
                "fraud_prediction"
            )

            probability = result.get(
                "fraud_probability"
            )

            threshold = result.get(
                "decision_threshold"
            )

            st.subheader(
                "Model result"
            )

            if prediction == 1:
                st.error(
                    "Potentially fraudulent — send for review"
                )
            else:
                st.success(
                    "Likely legitimate transaction"
                )

            if isinstance(
                probability,
                (int, float),
            ):

                probability = float(
                    probability
                )

                st.metric(
                    "Estimated fraud probability",
                    f"{probability:.1%}",
                )

                st.progress(
                    min(
                        max(
                            probability,
                            0.0,
                        ),
                        1.0,
                    )
                )

            if isinstance(
                threshold,
                (int, float),
            ):

                st.caption(
                    f"Decision threshold: "
                    f"{float(threshold):.1%}."
                )

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Dataset:** "
                    f"{result.get('dataset', dataset_name)}"
                )

                st.write(
                    f"**Model:** "
                    f"{result.get('classifier', 'Unknown')}"
                )

            with col2:
                st.write(
                    f"**Model version:** "
                    f"{result.get('model_version', model_version)}"
                )

                st.write(
                    f"**API schema:** "
                    f"{result.get('schema_version', 'Unknown')}"
                )

            if selected_ground_truth is not None:

                with st.expander(
                    "Compare with benchmark ground truth"
                ):

                    actual_label = (
                        "Fraud"
                        if selected_ground_truth == 1
                        else "Legitimate"
                    )

                    st.write(
                        f"**Actual dataset label:** "
                        f"{actual_label}"
                    )

                    if (
                        prediction
                        == selected_ground_truth
                    ):
                        st.success(
                            "Prediction matches the benchmark label."
                        )
                    else:
                        st.warning(
                            "Prediction does not match the benchmark label."
                        )

            with st.expander(
                "Prediction details"
            ):
                st.json(result)