from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# =========================================================
# FRAUDPULSE - CALIBRATED AI RISK ENGINE
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

OUT.mkdir(exist_ok=True)

FILE = DATA / "train_transaction.csv"

FEATURES = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "C1",
    "C2",
    "C5",
    "C13",
    "C14",
    "D1",
    "D2",
    "D4",
    "D10",
    "D15",
    "V12",
    "V13",
    "V14",
    "V17",
    "V34",
    "V35",
    "V36"
]

TARGET = "isFraud"

print("\nLoading transaction data...")

df = pd.read_csv(
    FILE,
    usecols=[TARGET] + FEATURES,
    dtype={
        "TransactionDT": "int32",
        "isFraud": "int8"
    }
)

df = df.sort_values(
    "TransactionDT"
).reset_index(drop=True)

print("Transactions:", len(df))

# =========================================================
# CHRONOLOGICAL SPLIT
# =========================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[
    train_end:validation_end
]
test = df.iloc[
    validation_end:
]

print("\nDataset split:")
print("Training:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

# =========================================================
# TRAIN BASE MODEL
# =========================================================

print("\nTraining fraud model...")

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=100,
    solver="liblinear",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model trained.")

# =========================================================
# RAW PROBABILITIES
# =========================================================

print("\nGenerating probabilities...")

val_raw = model.predict_proba(
    X_val
)[:, 1]

test_raw = model.predict_proba(
    X_test
)[:, 1]

print(
    "Raw validation probability range:",
    round(val_raw.min(), 4),
    "to",
    round(val_raw.max(), 4)
)

# =========================================================
# CALIBRATION
# =========================================================
#
# The balanced classifier deliberately changes class weighting.
# Therefore raw probabilities are not necessarily calibrated
# to the real-world fraud prevalence.
#
# Isotonic calibration is fitted ONLY on validation data.
# Test data remains completely held out.
# =========================================================

print("\nCalibrating fraud probabilities...")

calibrator = IsotonicRegression(
    y_min=0.0001,
    y_max=0.9999,
    out_of_bounds="clip"
)

calibrator.fit(
    val_raw,
    y_val
)

test_probability = calibrator.predict(
    test_raw
)

print(
    "Calibrated test probability range:",
    round(test_probability.min(), 4),
    "to",
    round(test_probability.max(), 4)
)

# =========================================================
# TEST FRAME
# =========================================================

risk_data = test[
    [
        "TransactionDT",
        "TransactionAmt",
        TARGET
    ]
].copy()

risk_data[
    "fraud_probability"
] = test_probability

risk_data[
    "transaction_risk"
] = (
    test_probability * 100
)

risk_data["hour_bin"] = (
    risk_data["TransactionDT"] // 3600
).astype("int32")

# =========================================================
# HOURLY RISK
# =========================================================

print("\nCreating hourly risk windows...")

hourly = (
    risk_data
    .groupby("hour_bin")
    .agg(
        transactions=(
            "fraud_probability",
            "size"
        ),

        average_risk=(
            "fraud_probability",
            "mean"
        ),

        observed_fraud=(
            TARGET,
            "sum"
        )
    )
    .reset_index()
)

# =========================================================
# HISTORICAL BASELINE
# =========================================================

print(
    "Calculating historical baseline..."
)

hourly["risk_baseline"] = (
    hourly[
        "average_risk"
    ]
    .shift(1)
    .rolling(
        window=24,
        min_periods=6
    )
    .mean()
)

hourly["risk_std"] = (
    hourly[
        "average_risk"
    ]
    .shift(1)
    .rolling(
        window=24,
        min_periods=6
    )
    .std()
)

hourly["risk_std"] = (
    hourly["risk_std"]
    .fillna(0)
)

effective_std = np.maximum(
    hourly["risk_std"].to_numpy(),
    0.002
)

# =========================================================
# RISK LIFT
# =========================================================

hourly["risk_lift"] = (
    hourly["average_risk"]
    /
    hourly["risk_baseline"]
)

hourly["risk_lift"] = (
    hourly["risk_lift"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

# =========================================================
# Z SCORE
# =========================================================

hourly["z_score"] = (
    (
        hourly["average_risk"]
        -
        hourly["risk_baseline"]
    )
    /
    effective_std
)

# =========================================================
# SPIKE SCORE
# =========================================================

z_component = np.clip(
    (
        hourly["z_score"] - 1
    ) / 4,
    0,
    1
)

lift_component = np.clip(
    (
        hourly["risk_lift"] - 1
    ) / 4,
    0,
    1
)

volume_component = np.clip(
    np.log1p(
        hourly["transactions"]
    )
    /
    np.log1p(1000),
    0,
    1
)

hourly["spike_score"] = (
    100 *
    (
        0.50 * z_component
        +
        0.30 * lift_component
        +
        0.20 * volume_component
    )
)

# =========================================================
# TRANSACTION RISK
# =========================================================

hourly["transaction_risk_score"] = (
    hourly["average_risk"] * 100
).clip(0, 100)

# =========================================================
# FINAL RISK
# =========================================================

hourly["final_risk_score"] = (
    0.60 *
    hourly["transaction_risk_score"]
    +
    0.40 *
    hourly["spike_score"].fillna(0)
)

hourly["final_risk_score"] = (
    hourly["final_risk_score"]
    .clip(0, 100)
)

# =========================================================
# SEVERITY
# =========================================================

hourly["severity"] = pd.cut(
    hourly["final_risk_score"],
    bins=[
        -1,
        30,
        60,
        80,
        100
    ],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]
)

# =========================================================
# EXPLANATIONS
# =========================================================

def explain(row):

    reasons = []

    if row["average_risk"] >= 0.10:

        reasons.append(
            "elevated predicted fraud probability"
        )

    if (
        pd.notna(row["risk_lift"])
        and
        row["risk_lift"] >= 1.5
    ):

        reasons.append(
            f"risk {row['risk_lift']:.1f}x "
            "above recent baseline"
        )

    if (
        pd.notna(row["z_score"])
        and
        row["z_score"] >= 2
    ):

        reasons.append(
            f"{row['z_score']:.1f}σ "
            "statistical deviation"
        )

    if row["transactions"] >= 100:

        reasons.append(
            f"{int(row['transactions'])} "
            "transactions in window"
        )

    if not reasons:

        reasons.append(
            "no major anomaly detected"
        )

    return "; ".join(reasons)


hourly["explanation"] = (
    hourly.apply(
        explain,
        axis=1
    )
)

# =========================================================
# ALERT POLICY
# =========================================================
#
# Slightly more sensitive than the first version.
# We require meaningful volume and either:
#
#   1. high final risk
# OR
#   2. a significant temporal anomaly.
#
# =========================================================

MIN_TRANSACTIONS = 100

hourly["alert"] = (

    (hourly["transactions"] >= MIN_TRANSACTIONS)

    &

    (
        (hourly["final_risk_score"] >= 50)

        |

        (
            (hourly["risk_lift"] >= 1.5)
            &
            (hourly["z_score"] >= 2)
        )
    )
)

alerts = hourly[
    hourly["alert"]
].copy()

alerts = alerts.sort_values(
    "final_risk_score",
    ascending=False
)

# =========================================================
# RETROSPECTIVE EVALUATION
# =========================================================

overall_fraud_rate = (
    y_test.mean()
)

if len(alerts) > 0:

    alert_fraud_rate = (
        alerts["observed_fraud"].sum()
        /
        alerts["transactions"].sum()
    )

else:

    alert_fraud_rate = 0

if overall_fraud_rate > 0:

    alert_lift = (
        alert_fraud_rate
        /
        overall_fraud_rate
    )

else:

    alert_lift = 0

# =========================================================
# SAVE
# =========================================================

hourly.to_csv(
    OUT /
    "risk_engine_hourly_results.csv",
    index=False
)

alerts.head(100).to_csv(
    OUT /
    "risk_engine_alerts.csv",
    index=False
)

summary = {

    "test_transactions":
        int(len(test)),

    "overall_test_fraud_rate":
        float(overall_fraud_rate),

    "alert_windows":
        int(len(alerts)),

    "alert_fraud_rate":
        float(alert_fraud_rate),

    "alert_fraud_lift":
        float(alert_lift),

    "calibration":
        "Isotonic regression fitted on validation data",

    "risk_formula":
        "60% transaction risk + 40% temporal spike risk",

    "label_usage":
        "isFraud used only for training/calibration and retrospective evaluation"
}

with open(
    OUT /
    "risk_engine_summary.json",
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

# =========================================================
# FINAL REPORT
# =========================================================

print("\n")
print("=" * 65)
print("              FRAUDPULSE RISK ENGINE")
print("=" * 65)

print(
    "\nTest transactions:",
    len(test)
)

print(
    "Overall fraud rate:",
    f"{overall_fraud_rate * 100:.2f}%"
)

print(
    "Alert windows:",
    len(alerts)
)

print(
    "Fraud rate in alert windows:",
    f"{alert_fraud_rate * 100:.2f}%"
)

print(
    "Alert fraud lift:",
    f"{alert_lift:.2f}x"
)

print("\nTOP RISK ALERTS")
print("-" * 65)

if len(alerts) == 0:

    print(
        "No alerts generated."
    )

else:

    columns = [
        "hour_bin",
        "transactions",
        "average_risk",
        "risk_lift",
        "z_score",
        "spike_score",
        "final_risk_score",
        "severity",
        "explanation"
    ]

    print(
        alerts[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

print("\nResults saved to:")

print(
    OUT /
    "risk_engine_hourly_results.csv"
)

print(
    OUT /
    "risk_engine_alerts.csv"
)

print(
    OUT /
    "risk_engine_summary.json"
)

print("\nRisk engine complete!")