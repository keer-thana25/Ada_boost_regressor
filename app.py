import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdaBoost Regression",
    page_icon="📈",
    layout="wide",
)

# ── Load dataset & train model (cached) ──────────────────────────────────────
@st.cache_resource
def load_and_train():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    base  = DecisionTreeRegressor(max_depth=3)
    model = AdaBoostRegressor(estimator=base, n_estimators=50, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    return data, X, y, X_test, y_test, scaler, model, y_pred

data, X, y, X_test, y_test, scaler, model, y_pred = load_and_train()
feature_names = data.feature_names

# ── Feature descriptions ──────────────────────────────────────────────────────
FEATURE_INFO = {
    "age":  ("Age of the patient (normalised).",
              "Each value is a z-score. 0 = average age in dataset.",
              "-0.11 – 0.11"),
    "sex":  ("Biological sex of the patient (normalised).",
              "Encoded as a continuous normalised value.",
              "-0.04 – 0.05"),
    "bmi":  ("Body Mass Index — weight relative to height.",
              "Higher BMI is linked to higher diabetes risk.",
              "-0.09 – 0.17"),
    "bp":   ("Average blood pressure (normalised).",
              "Higher blood pressure is associated with worse outcomes.",
              "-0.11 – 0.13"),
    "s1":   ("Total serum cholesterol (tc).",
              "One of six blood-serum measurements.",
              "-0.13 – 0.15"),
    "s2":   ("Low-density lipoprotein (ldl) — 'bad' cholesterol.",
              "Higher LDL is a diabetes risk factor.",
              "-0.12 – 0.20"),
    "s3":   ("High-density lipoprotein (hdl) — 'good' cholesterol.",
              "Higher HDL is generally protective.",
              "-0.10 – 0.18"),
    "s4":   ("Total cholesterol / HDL ratio (tch).",
              "A combined measure of cholesterol balance.",
              "-0.08 – 0.18"),
    "s5":   ("Log of serum triglycerides level (ltg).",
              "High triglycerides are linked to metabolic syndrome.",
              "-0.13 – 0.19"),
    "s6":   ("Blood sugar level (glu).",
              "Directly related to diabetes progression.",
              "-0.11 – 0.13"),
}

# ── Compute metrics ───────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 AdaBoost Regressor")
    st.markdown("**Dataset:** Diabetes (sklearn)")
    st.markdown("---")
    st.markdown(f"**Total samples :** {len(X)}")
    st.markdown(f"**Features      :** {X.shape[1]}")
    st.markdown(f"**Target        :** Disease progression score")
    st.markdown("---")
    st.metric("MAE",  f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R²",   f"{r2:.4f}")
    st.markdown("---")
    section = st.radio(
        "Navigate",
        ["📖 What is AdaBoost?", "📊 AdaBoost Regression", "🔮 Predict"],
    )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – WHAT IS ADABOOST?
# ═════════════════════════════════════════════════════════════════════════════
if section == "📖 What is AdaBoost?":
    st.title("📖 What is AdaBoost?")
    st.markdown("---")

    st.subheader("Simple Explanation")
    st.info(
        "**AdaBoost** (Adaptive Boosting) builds a strong predictor by combining many weak "
        "predictors in sequence. Each new predictor focuses on the samples where the "
        "previous one performed worst — adapting to its predecessor's mistakes."
    )

    st.subheader("Step-by-step: How Boosting Works")
    st.markdown(
        """
1. **Initialise** – All training samples start with equal weight.
2. **Train a weak model** (e.g., a shallow decision tree).
3. **Measure errors** – Samples with large residuals get higher weight.
4. **Train the next model** – It focuses more on those harder samples.
5. **Repeat** for N rounds (estimators).
6. **Combine** all models using a weighted average → final prediction.
        """
    )

    st.subheader("AdaBoost for Regression")
    st.success(
        "For regression tasks AdaBoost uses the **AdaBoost.R2** algorithm. "
        "Each weak regressor's contribution is weighted by its accuracy "
        "(lower error = higher weight). The final prediction is a weighted median "
        "of all weak regressors' outputs."
    )

    st.subheader("Core Equation")
    st.latex(r"\hat{y}(x) = \sum_{t=1}^{T} \alpha_t \, h_t(x)")
    st.caption(
        "hₜ(x) = t-th weak regressor   |   "
        "αₜ = its weight (inversely proportional to its error)   |   "
        "ŷ(x) = final predicted value"
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**✅ Advantages**")
        st.markdown("- Easy to tune\n- Works well on tabular data\n- Less prone to overfitting than single trees")
    with col2:
        st.markdown("**⚠️ Limitations**")
        st.markdown("- Sensitive to outliers\n- Slower than single models\n- Sequential training")
    with col3:
        st.markdown("**🌍 Real-world Uses**")
        st.markdown("- Medical outcome prediction\n- Housing price estimation\n- Energy consumption forecasting")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DATASET OVERVIEW & MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif section == "📊 AdaBoost Regression":
    st.title("📊 AdaBoost Regression – Diabetes Dataset")
    st.markdown("---")

    # ── Dataset overview ─────────────────────────────────────────────────────
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(X))
    c2.metric("Features", X.shape[1])
    c3.metric("Target Min", f"{y.min():.0f}")
    c4.metric("Target Max", f"{y.max():.0f}")

    st.caption(
        "The **target** is a quantitative measure of diabetes disease progression "
        "one year after baseline (range ≈ 25 – 346)."
    )

    st.markdown("**First 5 rows of the dataset**")
    st.dataframe(X.head(), use_container_width=True)

    st.markdown("**All 10 feature columns**")
    feat_df = pd.DataFrame({
        "Feature":       feature_names,
        "Meaning":       [FEATURE_INFO[f][1] for f in feature_names],
        "Description":   [FEATURE_INFO[f][0] for f in feature_names],
        "Typical Range": [FEATURE_INFO[f][2] for f in feature_names],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with st.expander("📋 Basic Statistics"):
        st.dataframe(X.describe(), use_container_width=True)

    st.markdown("---")

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.subheader("Model Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",  f"{mae:.2f}")
    m2.metric("MSE",  f"{mse:.2f}")
    m3.metric("RMSE", f"{rmse:.2f}")
    m4.metric("R²",   f"{r2:.4f}")

    # Actual vs Predicted
    st.markdown("**Actual vs Predicted**")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.set_style("whitegrid")
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.4)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # Residual plot
    st.markdown("**Residual Plot**")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.set_style("whitegrid")
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.4)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    st.pyplot(fig2)
    plt.close()

    # Feature importance
    st.markdown("**Feature Importances**")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.set_style("whitegrid")
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax3)
    ax3.set_title("Feature Importances")
    ax3.set_xlabel("Importance Score")
    st.pyplot(fig3)
    plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.title("🔮 Predict – Diabetes Disease Progression")
    st.markdown("---")
    st.info(
        "Adjust the sliders below to enter patient measurements. "
        "The model will predict the **disease progression score** one year after baseline."
    )

    user_input = {}
    left, right = st.columns(2)
    for i, feat in enumerate(feature_names):
        short_desc, long_desc, rng = FEATURE_INFO[feat]
        lo  = float(X[feat].min())
        hi  = float(X[feat].max())
        val = float(X[feat].mean())
        col = left if i % 2 == 0 else right
        with col:
            user_input[feat] = st.slider(
                label=feat,
                min_value=round(lo,  4),
                max_value=round(hi,  4),
                value=round(val, 4),
                help=f"{long_desc}  |  Typical range: {rng}",
            )
            st.caption(f"_{short_desc}_ — Range: {rng}")

    st.markdown("---")
    if st.button("🔮 Predict", use_container_width=True):
        input_df = pd.DataFrame([user_input])
        input_sc = scaler.transform(input_df)
        pred_val = model.predict(input_sc)[0]

        st.success(f"### 📊 Predicted Disease Progression Score: **{pred_val:.2f}**")

        # Interpret the score
        if pred_val < 100:
            level = "🟢 Low progression"
        elif pred_val < 200:
            level = "🟡 Moderate progression"
        else:
            level = "🔴 High progression"

        ca, cb = st.columns(2)
        ca.metric("Predicted Score", f"{pred_val:.2f}")
        cb.metric("Risk Level", level)

        st.caption(
            "Score range in dataset: ~25 (low risk) → ~346 (high risk). "
            "A lower score indicates less disease progression."
        )