from pathlib import Path
import json
import numpy as np
import pandas as pd

# =========================================================
# FraudPulse - AI Fraud Spike Detector
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

OUT.mkdir(exist_ok=True)

FILE = DATA / "train_transaction.csv"

print("\nLoading transaction data...")

# Only load the 3 columns we need.
df = pd.read_csv(
    FILE,
    usecols=[
        "TransactionDT",
        "TransactionAmt",
        "isFraud"
    ],
    dtype={
        "TransactionDT": "int32",
        "TransactionAmt": "float32",
        "isFraud": "int8"
    }
)

print("Dataset loaded.")
print("Transactions:", len(df))
print(
    "Overall fraud rate:",
    round(df["isFraud"].mean() * 100, 2),
    "%"
)

# =========================================================
# STEP 1: Create hourly windows
# =========================================================

print("\nCreating hourly windows...")

df["hour_bin"] = (
    df["TransactionDT"] // 3600
).astype("int32")

hourly = (
    df
    .groupby("hour_bin")
    .agg(
        transactions=("isFraud", "size"),
        fraud=("isFraud", "sum"),
        total_amount=("TransactionAmt", "sum")
    )
    .reset_index()
)

hourly["fraud_rate"] = (
    hourly["fraud"] /
    hourly["transactions"]
)

print(
    "Hourly windows:",
    len(hourly)
)

# =========================================================
# STEP 2: Historical baseline
# =========================================================

print("\nBuilding historical baseline...")

# IMPORTANT:
# shift(1) means the current hour is NOT included
# in its own baseline.

hourly["baseline_rate"] = (
    hourly["fraud_rate"]
    .shift(1)
    .rolling(
        window=24,
        min_periods=12
    )
    .mean()
)

# Historical standard deviation

hourly["baseline_std"] = (
    hourly["fraud_rate"]
    .shift(1)
    .rolling(
        window=24,
        min_periods=12
    )
    .std()
)

# =========================================================
# STEP 3: Protect against unstable statistics
# =========================================================

hourly["baseline_std"] = (
    hourly["baseline_std"]
    .fillna(0)
)

# Minimum standard deviation.

STD_FLOOR = 0.01

hourly["effective_std"] = np.maximum(
    hourly["baseline_std"],
    STD_FLOOR
)

# =========================================================
# STEP 4: Calculate fraud-rate lift
# =========================================================

hourly["rate_lift"] = (
    hourly["fraud_rate"] /
    hourly["baseline_rate"]
)

hourly["rate_lift"] = (
    hourly["rate_lift"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

# =========================================================
# STEP 5: Calculate statistical deviation
# =========================================================

hourly["z_score"] = (
    (
        hourly["fraud_rate"]
        -
        hourly["baseline_rate"]
    )
    /
    hourly["effective_std"]
)

# =========================================================
# STEP 6: Volume confidence
# =========================================================

hourly["volume_confidence"] = np.clip(
    np.log1p(hourly["transactions"])
    /
    np.log1p(1000),
    0,
    1
)

# =========================================================
# STEP 7: Risk score
# =========================================================

# Normalize statistical deviation.

z_component = np.clip(
    (hourly["z_score"] - 1) / 4,
    0,
    1
)

# Normalize rate increase.

lift_component = np.clip(
    (hourly["rate_lift"] - 1) / 4,
    0,
    1
)

# Combine signals.

hourly["risk_score"] = (
    100 *
    (
        0.50 * z_component
        +
        0.30 * lift_component
        +
        0.20 * hourly["volume_confidence"]
    )
)

# =========================================================
# STEP 8: Minimum transaction requirement
# =========================================================

MIN_TRANSACTIONS = 100

# Reduce confidence in very small windows.

hourly.loc[
    hourly["transactions"] < MIN_TRANSACTIONS,
    "risk_score"
] *= 0.35

hourly["risk_score"] = (
    hourly["risk_score"]
    .clip(0, 100)
)

# =========================================================
# STEP 9: Severity
# =========================================================

hourly["severity"] = pd.cut(
    hourly["risk_score"],
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
# STEP 10: Generate alerts
# =========================================================

hourly["alert"] = (
    (hourly["transactions"] >= MIN_TRANSACTIONS)
    &
    (
        hourly["fraud_rate"]
        >
        hourly["baseline_rate"] * 1.5
    )
    &
    (hourly["z_score"] >= 2)
)

alerts = hourly[
    hourly["alert"]
].copy()

alerts = alerts.sort_values(
    "risk_score",
    ascending=False
)

# =========================================================
# STEP 11: Save complete analysis
# =========================================================

hourly.to_csv(
    OUT / "hourly_spike_analysis.csv",
    index=False
)

alerts.head(100).to_csv(
    OUT / "top_spike_alerts.csv",
    index=False
)

# =========================================================
# STEP 12: Create summary
# =========================================================

summary = {
    "total_transactions": int(len(df)),

    "overall_fraud_rate":
        float(df["isFraud"].mean()),

    "hourly_windows":
        int(len(hourly)),

    "historical_window_hours":
        24,

    "minimum_transactions":
        MIN_TRANSACTIONS,

    "alert_windows":
        int(len(alerts))
}

with open(
    OUT / "spike_summary.json",
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
print("=" * 60)
print("             FRAUDPULSE SPIKE DETECTOR")
print("=" * 60)

print(
    "\nTotal transactions:",
    len(df)
)

print(
    "Hourly windows:",
    len(hourly)
)

print(
    "Overall fraud rate:",
    f"{df['isFraud'].mean() * 100:.2f}%"
)

print(
    "Minimum transactions:",
    MIN_TRANSACTIONS
)

print(
    "Alert windows:",
    len(alerts)
)

print("\nTOP FRAUD SPIKE ALERTS")
print("-" * 60)

if len(alerts) == 0:

    print(
        "No alerts met the detection criteria."
    )

else:

    columns = [
        "hour_bin",
        "transactions",
        "fraud",
        "fraud_rate",
        "baseline_rate",
        "rate_lift",
        "z_score",
        "risk_score",
        "severity"
    ]

    print(
        alerts[columns]
        .head(10)
        .to_string(index=False)
    )

print("\nFiles created:")

print(
    OUT /
    "hourly_spike_analysis.csv"
)

print(
    OUT /
    "top_spike_alerts.csv"
)

print(
    OUT /
    "spike_summary.json"
)

print("\nSpike detector complete!")