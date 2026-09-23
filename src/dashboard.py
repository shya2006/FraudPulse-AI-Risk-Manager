from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# FRAUDPULSE — AI RISK MANAGER
# Razorpay Buildathon Prototype
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="FraudPulse | AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f8fa;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 24px 28px;
    border-radius: 16px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 4px;
}

.hero p {
    color: #d1d5db;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.metric-title {
    color: #6b7280;
    font-size: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}

.metric-sub {
    color: #6b7280;
    font-size: 12px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

.alert-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: white;
    margin-bottom: 12px;
}

.small-text {
    font-size: 13px;
    color: #6b7280;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    return f"₹{value:,.2f}"


def risk_label(score):

    if score >= 80:
        return "CRITICAL"

    elif score >= 60:
        return "HIGH"

    elif score >= 30:
        return "MEDIUM"

    return "LOW"


def risk_action(score):

    if score >= 80:
        return (
            "HOLD / MANUAL REVIEW",
            "Temporarily hold the transaction flow and "
            "send the event for enhanced manual review."
        )

    elif score >= 60:
        return (
            "STEP-UP VERIFICATION",
            "Require additional verification and "
            "increase monitoring."
        )

    elif score >= 30:
        return (
            "MONITOR",
            "Continue processing while increasing "
            "monitoring and verification."
        )

    return (
        "ALLOW",
        "Continue normal processing."
    )


# ============================================================
# LOAD FILES
# ============================================================

risk_file = RESULTS / "risk_engine_hourly_results.csv"
alerts_file = RESULTS / "risk_engine_alerts.csv"
threshold_file = RESULTS / "threshold_analysis.csv"
summary_file = RESULTS / "risk_engine_summary.json"

if not risk_file.exists():

    st.error(
        "Risk engine results were not found."
    )

    st.code(
        "python src\\risk_engine.py"
    )

    st.stop()


df = pd.read_csv(risk_file)


if alerts_file.exists():

    alerts = pd.read_csv(alerts_file)

else:

    alerts = pd.DataFrame()


if threshold_file.exists():

    threshold_df = pd.read_csv(threshold_file)

else:

    threshold_df = pd.DataFrame()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

df.columns = [
    str(c).strip()
    for c in df.columns
]

if not alerts.empty:

    alerts.columns = [
        str(c).strip()
        for c in alerts.columns
    ]


# ============================================================
# CALCULATE DASHBOARD METRICS
# ============================================================

if "transactions" in df.columns:

    total_transactions = int(
        df["transactions"].sum()
    )

else:

    total_transactions = 0


if "observed_fraud" in df.columns:

    total_fraud = int(
        df["observed_fraud"].sum()
    )

else:

    total_fraud = 0


fraud_rate = (
    total_fraud / total_transactions
    if total_transactions > 0
    else 0
)


if not alerts.empty and "transactions" in alerts.columns:

    alert_transactions = int(
        alerts["transactions"].sum()
    )

    alert_fraud = int(
        alerts["observed_fraud"].sum()
    ) if "observed_fraud" in alerts.columns else 0

else:

    alert_transactions = 0
    alert_fraud = 0


alert_fraud_rate = (
    alert_fraud / alert_transactions
    if alert_transactions > 0
    else 0
)


fraud_lift = (
    alert_fraud_rate / fraud_rate
    if fraud_rate > 0
    else 0
)


if "final_risk_score" in df.columns:

    maximum_risk = float(
        df["final_risk_score"].max()
    )

else:

    maximum_risk = 0


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🛡️ FraudPulse</h1>

<p>
AI Risk Manager — Defense-only fraud detection,
temporal anomaly monitoring and explainable risk actions.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ FraudPulse")

st.sidebar.caption(
    "AI Risk Manager"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Risk Alerts",
        "Transaction Investigation",
        "Threshold Analysis",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Defense-only system. "
    "Designed to detect and reduce financial loss."
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Risk Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Transactions</div>
                <div class="metric-value">
                    {total_transactions:,}
                </div>
                <div class="metric-sub">
                    Evaluated transaction windows
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Observed Fraud Rate</div>
                <div class="metric-value">
                    {fraud_rate * 100:.2f}%
                </div>
                <div class="metric-sub">
                    Historical test-set rate
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Risk Alert Windows</div>
                <div class="metric-value">
                    {len(alerts):,}
                </div>
                <div class="metric-sub">
                    Statistically elevated periods
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Fraud Lift</div>
                <div class="metric-value">
                    {fraud_lift:.2f}×
                </div>
                <div class="metric-sub">
                    Fraud concentration in alerts
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ========================================================
    # RISK TIMELINE
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Risk Activity Timeline</div>',
        unsafe_allow_html=True
    )

    chart_columns = []

    if "final_risk_score" in df.columns:
        chart_columns.append("final_risk_score")

    if "spike_score" in df.columns:
        chart_columns.append("spike_score")

    if chart_columns:

        timeline = df[
            ["hour_bin"] + chart_columns
        ].copy()

        timeline = timeline.set_index(
            "hour_bin"
        )

        st.line_chart(
            timeline,
            height=380
        )

    # ========================================================
    # FRAUD RATE VS BASELINE
    # ========================================================

    if (
        "fraud_rate" in df.columns
        and "baseline_rate" in df.columns
    ):

        st.markdown(
            '<div class="section-title">'
            'Fraud Rate vs Historical Baseline'
            '</div>',
            unsafe_allow_html=True
        )

        chart = df[
            [
                "hour_bin",
                "fraud_rate",
                "baseline_rate"
            ]
        ].copy()

        chart["fraud_rate"] *= 100
        chart["baseline_rate"] *= 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart["hour_bin"],
                y=chart["fraud_rate"],
                mode="lines",
                name="Observed Fraud %"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart["hour_bin"],
                y=chart["baseline_rate"],
                mode="lines",
                name="Historical Baseline %"
            )
        )

        fig.update_layout(
            height=350,
            xaxis_title="Hour Window",
            yaxis_title="Fraud Rate (%)",
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# RISK ALERTS
# ============================================================

elif page == "Risk Alerts":

    st.header("🚨 Risk Alerts")

    if alerts.empty:

        st.success(
            "No active risk alerts."
        )

    else:

        # Sort highest risk first

        if "final_risk_score" in alerts.columns:

            alerts_display = alerts.sort_values(
                "final_risk_score",
                ascending=False
            ).copy()

        else:

            alerts_display = alerts.copy()


        for _, row in alerts_display.iterrows():

            score = float(
                row.get(
                    "final_risk_score",
                    0
                )
            )

            severity = risk_label(
                score
            )

            action, action_description = risk_action(
                score
            )

            st.markdown(
                f"""
                <div class="alert-box">

                <h3>
                🚨 {severity}
                — Risk Score {score:.1f}/100
                </h3>

                <p>
                <b>Hour window:</b>
                {row.get("hour_bin", "N/A")}
                &nbsp;&nbsp;

                <b>Transactions:</b>
                {int(row.get("transactions", 0))}

                &nbsp;&nbsp;

                <b>Risk Lift:</b>
                {float(row.get("risk_lift", 0)):.2f}×

                &nbsp;&nbsp;

                <b>Z-Score:</b>
                {float(row.get("z_score", 0)):.2f}σ
                </p>

                <p>
                <b>Recommended Action:</b>
                {action}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                action_description
            )

            explanation = row.get(
                "explanation",
                "Elevated risk detected."
            )

            st.write(
                f"**Why was this flagged?** {explanation}"
            )

            st.divider()


# ============================================================
# TRANSACTION INVESTIGATION
# ============================================================

elif page == "Transaction Investigation":

    st.header("🔎 Transaction Investigation")

    st.write(
        "Investigate risk windows using the available "
        "model and temporal signals."
    )

    if df.empty:

        st.warning(
            "No transaction risk data available."
        )

    else:

        # Create transaction/window selector

        if "hour_bin" in df.columns:

            selected_hour = st.selectbox(
                "Select risk window",
                df["hour_bin"].tolist()
            )

            selected = df[
                df["hour_bin"]
                == selected_hour
            ].iloc[0]

        else:

            selected = df.iloc[0]


        score = float(
            selected.get(
                "final_risk_score",
                0
            )
        )

        severity = risk_label(
            score
        )

        action, description = risk_action(
            score
        )

        st.divider()

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Risk Score",
                f"{score:.1f}/100"
            )

        with c2:

            st.metric(
                "Average Risk",
                f"{float(selected.get('average_risk', 0)) * 100:.2f}%"
            )

        with c3:

            st.metric(
                "Risk Lift",
                f"{float(selected.get('risk_lift', 0)):.2f}×"
            )

        with c4:

            st.metric(
                "Z-Score",
                f"{float(selected.get('z_score', 0)):.2f}σ"
            )

        st.divider()

        st.subheader(
            f"Risk Level: {severity}"
        )

        st.warning(
            f"Recommended action: **{action}**\n\n"
            f"{description}"
        )

        st.subheader(
            "Risk Signals"
        )

        signals = []

        if float(selected.get("risk_lift", 0)) > 1:

            signals.append(
                f"Risk is "
                f"{float(selected.get('risk_lift', 0)):.2f}× "
                f"above the historical baseline."
            )

        if float(selected.get("z_score", 0)) >= 3:

            signals.append(
                f"Statistically unusual activity: "
                f"{float(selected.get('z_score', 0)):.2f}σ."
            )

        if float(selected.get("spike_score", 0)) >= 50:

            signals.append(
                "Significant temporal fraud spike detected."
            )

        transactions = int(
            selected.get(
                "transactions",
                0
            )
        )

        signals.append(
            f"{transactions:,} transactions "
            "occurred in this window."
        )

        for signal in signals:

            st.write(
                f"✓ {signal}"
            )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

elif page == "Threshold Analysis":

    st.header("⚖️ Threshold & Cost Analysis")

    st.write(
        "The system evaluates the trade-off between "
        "false positives and missed fraud."
    )

    if threshold_df.empty:

        st.warning(
            "Run threshold_analysis.py first."
        )

        st.code(
            "python src\\threshold_analysis.py"
        )

    else:

        # ====================================================
        # COST CHART
        # ====================================================

        st.subheader(
            "Business Cost by Threshold"
        )

        fig = px.line(
            threshold_df,
            x="threshold",
            y="total_cost",
            markers=True,
            labels={
                "threshold": "Decision Threshold",
                "total_cost": "Estimated Cost"
            }
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ====================================================
        # PRECISION / RECALL
        # ====================================================

        st.subheader(
            "Precision vs Recall"
        )

        metric_df = threshold_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1"
            ]
        ].copy()

        metric_long = metric_df.melt(
            id_vars="threshold",
            var_name="metric",
            value_name="value"
        )

        fig2 = px.line(
            metric_long,
            x="threshold",
            y="value",
            color="metric",
            markers=True
        )

        fig2.update_layout(
            height=400,
            yaxis_title="Score"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ====================================================
        # BEST THRESHOLD
        # ====================================================

        best = threshold_df.loc[
            threshold_df["total_cost"].idxmin()
        ]

        st.subheader(
            "Recommended Operating Point"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Threshold",
            f"{best['threshold']:.2f}"
        )

        c2.metric(
            "Precision",
            f"{best['precision'] * 100:.2f}%"
        )

        c3.metric(
            "Recall",
            f"{best['recall'] * 100:.2f}%"
        )

        c4.metric(
            "Estimated Cost",
            f"{int(best['total_cost']):,}"
        )

        st.info(
            "Cost assumptions used in this prototype: "
            "false positive = 1 unit, "
            "false negative = 10 units. "
            "These are demonstration assumptions, "
            "not Razorpay production costs."
        )

        st.subheader(
            "Complete Threshold Table"
        )

        st.dataframe(
            threshold_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Held-out test-set performance from the FraudPulse "
        "fraud detection model."
    )

    # Known from latest threshold analysis

    metrics = {
        "Precision": 0.1469,
        "Recall": 0.4029,
        "F1 Score": 0.2153,
        "PR-AUC": 0.1180,
        "ROC-AUC": 0.7332
    }

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols,
        metrics.items()
    ):

        with col:

            st.metric(
                name,
                f"{value:.4f}"
            )

    st.divider()

    st.subheader(
        "Confusion Matrix — Held-out Test Set"
    )

    tp = 1242
    fp = 7210
    fn = 1841
    tn = 78288

    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[
                "Predicted Legitimate",
                "Predicted Fraud"
            ],
            y=[
                "Actual Legitimate",
                "Actual Fraud"
            ],
            text=matrix,
            texttemplate="%{text:,}",
            colorscale="Blues"
        )
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Interpretation"
    )

    st.markdown(
        """
        **Precision:** 14.69% of transactions flagged by the
        model were actually fraudulent.

        **Recall:** 40.29% of fraudulent transactions were
        detected at the selected threshold.

        **PR-AUC:** 0.1180, which is especially relevant for
        this imbalanced fraud-detection problem.

        **ROC-AUC:** 0.7332, indicating useful ranking
        capability between fraudulent and legitimate
        transactions.
        """
    )

    st.warning(
        "The model should be treated as a risk-ranking "
        "component, not an autonomous decision maker."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FraudPulse • AI Risk Manager • Razorpay Buildathon Prototype"
)

st.caption(
    "Defense-only • Explainable • Cost-aware • "
    "Evaluated on a held-out test set"
)