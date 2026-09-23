from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# =========================================================
# FRAUDPULSE - THRESHOLD & FALSE-POSITIVE COST ANALYSIS
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

# =========================================================
# BUSINESS COST ASSUMPTIONS
# =========================================================
#
# These are DEMO assumptions, not Razorpay's real costs.
#
# False positive:
# Legitimate transaction incorrectly flagged.
#
# False negative:
# Fraudulent transaction incorrectly allowed.
#
# Fraud cost is intentionally higher.
# =========================================================

FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 10

print("\nLoading dataset...")

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

print(
    f"Transactions: {len(df):,}"
)

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
print(
    f"Training:   {len(train):,}"
)

print(
    f"Validation: {len(validation):,}"
)

print(
    f"Test:       {len(test):,}"
)

# =========================================================
# PREPARE FEATURES
# =========================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("\nHandling missing values...")

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_val = imputer.transform(
    X_val
)

X_test = imputer.transform(
    X_test
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining model...")

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
# PREDICTIONS
# =========================================================

print(
    "\nGenerating validation/test probabilities..."
)

val_probability = model.predict_proba(
    X_val
)[:, 1]

test_probability = model.predict_proba(
    X_test
)[:, 1]

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

thresholds = np.arange(
    0.10,
    0.96,
    0.05
)

rows = []

print(
    "\nEvaluating thresholds..."
)

for threshold in thresholds:

    # Validation predictions
    val_pred = (
        val_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    # Business cost
    total_cost = (
        fp * FALSE_POSITIVE_COST
        +
        fn * FALSE_NEGATIVE_COST
    )

    rows.append({

        "threshold":
            round(float(threshold), 2),

        "precision":
            round(float(precision), 4),

        "recall":
            round(float(recall), 4),

        "f1":
            round(float(f1), 4),

        "true_positives":
            int(tp),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_negatives":
            int(tn),

        "false_positive_cost":
            int(
                fp *
                FALSE_POSITIVE_COST
            ),

        "false_negative_cost":
            int(
                fn *
                FALSE_NEGATIVE_COST
            ),

        "total_cost":
            int(total_cost)
    })

results = pd.DataFrame(
    rows
)

# =========================================================
# BEST THRESHOLDS
# =========================================================

best_cost_row = results.loc[
    results["total_cost"].idxmin()
]

best_f1_row = results.loc[
    results["f1"].idxmax()
]

best_recall_row = results.loc[
    results["recall"].idxmax()
]

# =========================================================
# TEST EVALUATION
# =========================================================

selected_threshold = float(
    best_cost_row["threshold"]
)

test_prediction = (
    test_probability >=
    selected_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
).ravel()

test_precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_pr_auc = average_precision_score(
    y_test,
    test_probability
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

test_cost = (
    fp * FALSE_POSITIVE_COST
    +
    fn * FALSE_NEGATIVE_COST
)

# =========================================================
# SAVE TABLE
# =========================================================

threshold_file = (
    OUT /
    "threshold_analysis.csv"
)

results.to_csv(
    threshold_file,
    index=False
)

# =========================================================
# SAVE SUMMARY
# =========================================================

summary = {

    "cost_assumptions": {

        "false_positive_cost":
            FALSE_POSITIVE_COST,

        "false_negative_cost":
            FALSE_NEGATIVE_COST,

        "note":
            "Demo assumptions for buildathon analysis; "
            "not Razorpay production costs."
    },

    "best_validation_threshold":
        selected_threshold,

    "best_validation_cost":
        int(
            best_cost_row["total_cost"]
        ),

    "best_validation_f1_threshold":
        float(
            best_f1_row["threshold"]
        ),

    "best_validation_f1":
        float(
            best_f1_row["f1"]
        ),

    "held_out_test": {

        "precision":
            float(test_precision),

        "recall":
            float(test_recall),

        "f1":
            float(test_f1),

        "pr_auc":
            float(test_pr_auc),

        "roc_auc":
            float(test_roc_auc),

        "true_positives":
            int(tp),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_negatives":
            int(tn),

        "estimated_business_cost":
            int(test_cost)
    }
}

summary_file = (
    OUT /
    "threshold_analysis_summary.json"
)

with open(
    summary_file,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

# =========================================================
# DISPLAY
# =========================================================

print("\n")
print("=" * 75)
print("           FRAUDPULSE THRESHOLD ANALYSIS")
print("=" * 75)

print(
    "\nFalse-positive cost:",
    FALSE_POSITIVE_COST
)

print(
    "False-negative cost:",
    FALSE_NEGATIVE_COST
)

print(
    "\nBEST VALIDATION THRESHOLD"
)

print(
    "Threshold:",
    selected_threshold
)

print(
    "Total cost:",
    int(
        best_cost_row["total_cost"]
    )
)

print(
    "Precision:",
    f"{best_cost_row['precision']:.4f}"
)

print(
    "Recall:",
    f"{best_cost_row['recall']:.4f}"
)

print(
    "\nHELD-OUT TEST PERFORMANCE"
)

print(
    "Precision:",
    f"{test_precision:.4f}"
)

print(
    "Recall:",
    f"{test_recall:.4f}"
)

print(
    "F1:",
    f"{test_f1:.4f}"
)

print(
    "PR-AUC:",
    f"{test_pr_auc:.4f}"
)

print(
    "ROC-AUC:",
    f"{test_roc_auc:.4f}"
)

print(
    "\nCONFUSION MATRIX"
)

print(
    "True Positives :",
    tp
)

print(
    "False Positives:",
    fp
)

print(
    "False Negatives:",
    fn
)

print(
    "True Negatives :",
    tn
)

print(
    "\nESTIMATED TEST COST:",
    test_cost
)

print(
    "\nTHRESHOLD COMPARISON"
)

print("-" * 75)

print(
    results[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positives",
            "false_negatives",
            "total_cost"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nFiles saved:"
)

print(
    threshold_file
)

print(
    summary_file
)

print(
    "\nThreshold analysis complete!"
)