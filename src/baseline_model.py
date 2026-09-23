from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ---------------------------------------------------------
# FraudPulse - Memory Efficient Baseline
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

OUT.mkdir(exist_ok=True)

FILE = DATA / "train_transaction.csv"


# ---------------------------------------------------------
# Only use compact, useful features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Load only required columns
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    FILE,
    usecols=[TARGET] + FEATURES,
    dtype={
        "isFraud": "int8",

        "TransactionAmt": "float32",
        "TransactionDT": "int32",

        "card1": "float32",
        "card2": "float32",
        "card3": "float32",
        "card5": "float32",

        "addr1": "float32",
        "addr2": "float32",

        "C1": "float32",
        "C2": "float32",
        "C5": "float32",
        "C13": "float32",
        "C14": "float32",

        "D1": "float32",
        "D2": "float32",
        "D4": "float32",
        "D10": "float32",
        "D15": "float32",

        "V12": "float32",
        "V13": "float32",
        "V14": "float32",
        "V17": "float32",
        "V34": "float32",
        "V35": "float32",
        "V36": "float32"
    }
)

print("Dataset loaded.")
print("Rows:", len(df))
print("Fraud rate:", round(df[TARGET].mean() * 100, 2), "%")


# ---------------------------------------------------------
# Sort chronologically
# ---------------------------------------------------------

print("\nSorting transactions by time...")

df = df.sort_values("TransactionDT")

df.reset_index(drop=True, inplace=True)


# ---------------------------------------------------------
# Chronological split
# ---------------------------------------------------------

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[train_end:validation_end]
test = df.iloc[validation_end:]

print("\nDataset split:")
print("-------------------------")
print("Training:", len(train))
print("Validation:", len(validation))
print("Held-out test:", len(test))

print("\nFraud rates:")
print("-------------------------")
print("Training:",
      round(train[TARGET].mean() * 100, 2), "%")

print("Validation:",
      round(validation[TARGET].mean() * 100, 2), "%")

print("Test:",
      round(test[TARGET].mean() * 100, 2), "%")


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ---------------------------------------------------------
# Median imputation
# ---------------------------------------------------------

print("\nHandling missing values...")

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)


# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------

print("\nTraining baseline Logistic Regression...")
print("This may take a few minutes.\n")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=100,
    solver="liblinear",
    random_state=42
)

model.fit(X_train, y_train)

print("Model training complete.")


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

print("\nGenerating predictions...")

validation_probability = model.predict_proba(X_val)[:, 1]

test_probability = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------

print("\nSelecting threshold using validation data...")

FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 10

best_threshold = 0.5
best_cost = float("inf")

for threshold in np.arange(0.05, 0.96, 0.05):

    validation_prediction = (
        validation_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        validation_prediction,
        labels=[0, 1]
    ).ravel()

    cost = (
        fp * FALSE_POSITIVE_COST
        +
        fn * FALSE_NEGATIVE_COST
    )

    if cost < best_cost:

        best_cost = cost
        best_threshold = threshold


print(
    "Selected threshold:",
    round(best_threshold, 2)
)


# ---------------------------------------------------------
# Final evaluation on future test set
# ---------------------------------------------------------

test_prediction = (
    test_probability >= best_threshold
).astype(int)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
).ravel()


precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

pr_auc = average_precision_score(
    y_test,
    test_probability
)

roc_auc = roc_auc_score(
    y_test,
    test_probability
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

results = {

    "model": "Memory-efficient Logistic Regression",

    "split":
        "70% chronological train / "
        "15% validation / "
        "15% future held-out test",

    "threshold": float(best_threshold),

    "precision": float(precision),

    "recall": float(recall),

    "f1": float(f1),

    "pr_auc": float(pr_auc),

    "roc_auc": float(roc_auc),

    "true_positive": int(tp),

    "false_positive": int(fp),

    "false_negative": int(fn),

    "true_negative": int(tn),

    "test_rows": int(len(test)),

    "test_fraud_rate":
        float(y_test.mean()),

    "false_positive_cost":
        FALSE_POSITIVE_COST,

    "false_negative_cost":
        FALSE_NEGATIVE_COST
}


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_file = OUT / "baseline_metrics.json"

with open(output_file, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ---------------------------------------------------------
# Print final report
# ---------------------------------------------------------

print("\n")
print("=" * 50)
print("           FRAUDPULSE BASELINE")
print("=" * 50)

print(
    f"\nPrecision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("\nConfusion Matrix")
print("-------------------------")

print("True Positives :", tp)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Negatives :", tn)

print("\nThreshold:", best_threshold)

print("\nResults saved to:")

print(output_file)

print("\nBaseline complete!")