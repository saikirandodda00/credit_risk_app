

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">💳 AI Credit Risk Scoring System</p>', unsafe_allow_html=True)
  




# Load model
model = joblib.load("credit_risk_model_small.pkl")

st.title("💳 Credit Risk Prediction App")

st.write("Enter customer details")

# -------- Numeric Inputs --------

EXT_SOURCE_1 = st.slider("External Score 1", 0.0, 1.0, 0.5)
EXT_SOURCE_2 = st.slider("External Score 2", 0.0, 1.0, 0.5)
EXT_SOURCE_3 = st.slider("External Score 3", 0.0, 1.0, 0.5)

AMT_CREDIT = st.number_input("Credit Amount", 10000, 2000000, 500000)

# -------- Categorical Inputs --------

NAME_INCOME_TYPE = st.selectbox(
    "Income Type",
    ["Working", "Pensioner", "Commercial associate", "State servant"]
)

NAME_EDUCATION_TYPE = st.selectbox(
    "Education Type",
    ["Secondary / secondary special", "Higher education"]
)

CODE_GENDER = st.selectbox(
    "Gender",
    ["M", "F"]
)

# -------- Create DataFrame --------

input_data = pd.DataFrame({
    "EXT_SOURCE_1": [EXT_SOURCE_1],
    "EXT_SOURCE_2": [EXT_SOURCE_2],
    "EXT_SOURCE_3": [EXT_SOURCE_3],
    "AMT_CREDIT": [AMT_CREDIT],
    "NAME_INCOME_TYPE": [NAME_INCOME_TYPE],
    "NAME_EDUCATION_TYPE": [NAME_EDUCATION_TYPE],
    "CODE_GENDER": [CODE_GENDER]
})

# -------- Prediction --------

if st.button("Predict"):
    probability = model.predict_proba(input_data)[:,1][0]
    
    st.subheader(f"Default Probability: {probability:.2%}")
    
    # Risk label
    if probability > 0.6:
        st.error("🔴 High Risk Customer")
    elif probability > 0.4:
        st.warning("🟠 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")

    # ---- SHAP Explanation ----
    st.subheader("🔎 Model Explanation")

    # Extract internal model
    xgb_model = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    input_transformed = preprocessor.transform(input_data)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_transformed)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_transformed[0]
        ),
        max_display=7
    )
    st.pyplot(fig)