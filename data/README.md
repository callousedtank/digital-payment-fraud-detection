# Dataset

The training dataset is not committed to the repository. Place it here as
`Digital_Payment_Fraud_Detection_Dataset.csv`.

The expected CSV has 7,500 transactions and these columns:

* Identifiers: `transaction_id`, `user_id` (excluded from model features)
* Features: `transaction_amount`, `transaction_type`, `payment_mode`, `device_type`, `device_location`, `account_age_days`, `transaction_hour`, `previous_failed_attempts`, `avg_transaction_amount`, `is_international`, `ip_risk_score`, and `login_attempts_last_24h`
* Target: `fraud_label` (binary: `0` legitimate, `1` fraudulent)

The preprocessing pipeline rejects missing values and requires both target classes. Record the dataset fingerprint written to `experiments/fraud-detection.jsonl` when comparing training runs.
