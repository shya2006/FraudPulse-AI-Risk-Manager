<<<<<<< HEAD
# 🛡️ FraudPulse — AI Risk Manager

### Razorpay Buildathon | AI Risk Manager Track

FraudPulse is a defense-only AI risk management system designed to help merchants identify unusual fraud activity, prioritize risky transaction periods, and recommend appropriate risk actions.

The system combines **transaction-level machine learning** with **temporal fraud-spike detection** and **cost-aware threshold selection**.

---

## 🎯 Problem

Fraud, returns, and chargebacks can quietly reduce merchant margins.

A useful risk system should not simply classify transactions as "fraud" or "legitimate." It should:

* identify suspicious transactions,
* detect sudden fraud spikes,
* quantify how unusual the activity is,
* balance fraud detection against false-positive cost,
* explain why an event was flagged,
* and recommend an appropriate defensive action.

---

# 💡 Solution

FraudPulse combines three major components:

### 1. Transaction Risk Model

A Logistic Regression model estimates the probability that a transaction is fraudulent.

Features include transaction amount, card-related variables, address information, transaction timing, and selected behavioral/transaction features.

### 2. Temporal Spike Detector

Fraud activity is aggregated into hourly windows.

For each window, FraudPulse calculates:

* observed fraud rate,
* historical baseline rate,
* risk lift,
* statistical z-score,
* spike score.

This allows the system to identify sudden changes in fraud activity rather than relying only on individual transactions.

### 3. Risk Engine

The risk engine combines transaction-level risk and temporal anomaly signals into an operational risk score.

The output is mapped to defensive actions:

| Risk Level | Recommended Action              |
| ---------- | ------------------------------- |
| LOW        | Allow and monitor               |
| MEDIUM     | Monitor / increase verification |
| HIGH       | Step-up verification            |
| CRITICAL   | Hold / manual review            |

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Transaction Data   │
                    │   590,540 Records    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Preparation   │
                    │ • Missing values     │
                    │ • Feature selection  │
                    │ • Time sorting       │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
       ┌─────────────────────┐   ┌─────────────────────┐
       │  Transaction ML      │   │ Temporal Detector   │
       │                     │   │                     │
       │ Logistic Regression │   │ Hourly aggregation  │
       │ Fraud probability   │   │ Historical baseline │
       │ Probability ranking │   │ Z-score             │
       └──────────┬──────────┘   │ Risk lift           │
                  │              │ Spike score         │
                  │              └──────────┬──────────┘
                  │                         │
                  └────────────┬────────────┘
                               ▼
                    ┌──────────────────────┐
                    │     Risk Engine      │
                    │                      │
                    │ Combined risk score  │
                    │ Alert generation     │
                    │ Explainable signals  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Risk Classification│
                    │                      │
                    │ LOW                  │
                    │ MEDIUM               │
                    │ HIGH                 │
                    │ CRITICAL             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Recommended Action   │
                    │                      │
                    │ Allow                │
                    │ Monitor              │
                    │ Step-up Verification │
                    │ Manual Review        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  FraudPulse Dashboard│
                    │                      │
                    │ Alerts               │
                    │ Risk Timeline        │
                    │ Investigation        │
                    │ Threshold Analysis   │
                    │ Model Metrics        │
                    └──────────────────────┘
```

---

# 📊 Dataset

The prototype uses the IEEE-CIS Fraud Detection transaction dataset.

Total transactions:

**590,540**

Chronological split:

| Dataset       | Records |
| ------------- | ------: |
| Training      | 413,378 |
| Validation    |  88,581 |
| Held-out Test |  88,581 |

Overall fraud rate in the held-out test set:

**3.48%**

The chronological split prevents future transactions from being used to train the model.

---

# 🤖 Machine Learning Model

The baseline fraud model uses Logistic Regression with class balancing.

The pipeline performs:

1. Feature selection
2. Missing-value imputation
3. Chronological splitting
4. Model training
5. Probability generation
6. Threshold analysis
7. Held-out test evaluation

The model is used primarily as a **risk-ranking component**.

---

# ⚖️ Threshold & Cost Analysis

Fraud detection involves a trade-off.

A very sensitive threshold can detect more fraud but may incorrectly flag many legitimate transactions.

A very conservative threshold reduces false positives but allows more fraud to pass.

FraudPulse evaluates multiple thresholds.

For the prototype, the following illustrative cost assumptions are used:

```text
False Positive Cost = 1
False Negative Cost = 10
```

These are **demonstration assumptions and are not Razorpay production costs**.

The minimum validation cost was observed at:

**Threshold = 0.75**

Validation results at this threshold:

* Precision: **15.18%**
* Recall: **38.43%**
* Estimated cost: **25,264**

---

# 🧪 Held-Out Test Results

At the selected threshold:

| Metric    |     Result |
| --------- | ---------: |
| Precision | **14.69%** |
| Recall    | **40.29%** |
| F1 Score  | **0.2153** |
| PR-AUC    | **0.1180** |
| ROC-AUC   | **0.7332** |

Confusion matrix:

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------: | --------------: |
| Actual Legitimate |               78,288 |           7,210 |
| Actual Fraud      |                1,841 |           1,242 |

These metrics are reported on a **held-out test set** rather than the training data.

---

# 🚨 Temporal Fraud Spike Detection

FraudPulse also detects unusual hourly activity.

The spike detector identified **29 anomalous hourly windows** in its analysis.

The risk engine identified a smaller set of higher-confidence risk windows.

The strongest risk-engine result showed:

* 111 transactions
* Average risk: **8.31%**
* Risk lift: **2.17×**
* Statistical deviation: **5.78σ**
* Final risk score: **33.95**

Across the identified risk alert windows:

**Observed fraud rate: 12.29%**

compared with:

**Overall test fraud rate: 3.48%**

This corresponds to approximately:

# **3.53× fraud lift**

This demonstrates that the temporal risk signal can concentrate attention on periods with substantially higher observed fraud.

---

# 🔍 Explainability

Every risk alert provides supporting signals such as:

* risk above historical baseline,
* statistical deviation,
* transaction volume,
* elevated fraud probability,
* temporal spike activity.

Example:

```text
Risk Lift: 2.17×
Z-Score: 5.78σ
Transactions: 111

Why flagged?

Risk is significantly above the recent baseline
and the activity represents a statistically unusual
fraud-risk window.
```

The system therefore provides an explanation alongside the risk score instead of producing an unexplained prediction.

---

# 🖥️ Dashboard

FraudPulse includes a Streamlit dashboard with:

### Overview

* transaction volume,
* fraud rate,
* alert count,
* fraud lift,
* risk timeline.

### Risk Alerts

* severity,
* risk score,
* risk lift,
* z-score,
* recommended action,
* explanation.

### Transaction Investigation

* risk-window selection,
* risk score,
* average risk,
* anomaly signals,
* recommended action.

### Threshold Analysis

* cost curve,
* precision/recall curve,
* operating threshold,
* threshold comparison table.

### Model Performance

* precision,
* recall,
* F1,
* PR-AUC,
* ROC-AUC,
* confusion matrix.

---

# 📁 Project Structure

```text
FraudPulse_Razorpay_Project/
│
├── data/
│   └── train_transaction.csv
│
├── src/
│   ├── baseline_model.py
│   ├── spike_detector.py
│   ├── risk_engine.py
│   ├── threshold_analysis.py
│   └── dashboard.py
│
├── results/
│   ├── baseline_metrics.json
│   ├── hourly_spike_analysis.csv
│   ├── top_spike_alerts.csv
│   ├── spike_summary.json
│   ├── risk_engine_hourly_results.csv
│   ├── risk_engine_alerts.csv
│   ├── risk_engine_summary.json
│   ├── threshold_analysis.csv
│   └── threshold_analysis_summary.json
│
├── requirements.txt
└── README.md
```

---

# ▶️ Running the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the fraud model:

```bash
python src/baseline_model.py
```

Run the spike detector:

```bash
python src/spike_detector.py
```

Run the risk engine:

```bash
python src/risk_engine.py
```

Run threshold analysis:

```bash
python src/threshold_analysis.py
```

Launch the dashboard:

```bash
python -m streamlit run src/dashboard.py
```

The dashboard will open locally in the browser.

---

# 🔐 Defense-Only Design

FraudPulse is designed strictly for defensive risk management.

It does not provide:

* fraud execution techniques,
* payment bypass methods,
* credential theft,
* exploitation instructions,
* attack automation.

Its purpose is to identify suspicious activity and recommend defensive controls.

---

# ⚠️ Limitations

### 1. Dataset limitation

The prototype uses a public fraud dataset rather than live merchant traffic.

### 2. Cost assumptions

False-positive and false-negative costs are illustrative and should be replaced with merchant-specific economic estimates in production.

### 3. Model performance

The model is not intended to eliminate fraud completely. It is a risk-ranking component.

### 4. Concept drift

Fraud patterns change over time. A production system would require continuous monitoring and retraining.

### 5. Alert calibration

Risk thresholds should be calibrated using real operational outcomes and merchant risk tolerance.

### 6. Production integration

The prototype does not directly connect to payment authorization, merchant systems, or production transaction infrastructure.

---

# 🚀 Future Improvements

Potential production extensions include:

* gradient-boosted fraud models,
* graph-based fraud-ring detection,
* merchant-specific risk models,
* online feature stores,
* real-time streaming,
* adaptive thresholds,
* feedback from confirmed fraud,
* chargeback/return intelligence,
* automated case management,
* model drift monitoring.

---

# 🏆 Key Takeaway

FraudPulse demonstrates a defense-first approach to fraud risk management by combining:

**Machine Learning + Temporal Anomaly Detection + Cost-Aware Decisions + Explainability**

The strongest prototype result is a **3.53× fraud lift in identified risk alert windows**, while performance is evaluated on a held-out test set.

The system is designed not merely to say:

> "This transaction looks fraudulent."

but to answer:

> **"How risky is this activity, why is it unusual, and what defensive action should be taken?"**

---

## Built for the Razorpay AI Risk Manager Track

**FraudPulse — Turning fraud signals into actionable risk decisions.**
=======
# FraudPulse-AI-Risk-Manager
FraudPulse is a defense-only AI risk management system designed to help merchants identify unusual fraud activity, prioritize risky transaction periods, and recommend appropriate risk actions.  The system combines transaction-level machine learning with temporal fraud-spike detection and cost-aware threshold selection.
>>>>>>> 34a5c12f18118e2a2fcfde5c71b4b79f1095563c
