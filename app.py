import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Credit Risk AI", page_icon="💳")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">💳 AI Credit Risk Scoring System</p>', unsafe_allow_html=True)

# ===============================
# Cached Model Loader
# ===============================

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model_small.pkl")

with st.spinner("Loading model..."):
    model = load_model()

st.write("Enter customer details")

# ===============================
# Inputs
# ===============================

EXT_SOURCE_1 = st.slider("External Score 1", 0.0, 1.0, 0.5)
EXT_SOURCE_2 = st.slider("External Score 2", 0.0, 1.0, 0.5)
EXT_SOURCE_3 = st.slider("External Score 3", 0.0, 1.0, 0.5)
AMT_CREDIT = st.number_input("Credit Amount", 10000, 2000000, 500000)

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

# ===============================
# Create Input DataFrame
# ===============================

input_data = pd.DataFrame({
    "EXT_SOURCE_1": [EXT_SOURCE_1],
    "EXT_SOURCE_2": [EXT_SOURCE_2],
    "EXT_SOURCE_3": [EXT_SOURCE_3],
    "AMT_CREDIT": [AMT_CREDIT],
    "NAME_INCOME_TYPE": [NAME_INCOME_TYPE],
    "NAME_EDUCATION_TYPE": [NAME_EDUCATION_TYPE],
    "CODE_GENDER": [CODE_GENDER]
})

# ===============================
# Prediction
# ===============================

if st.button("Predict"):

    probability = model.predict_proba(input_data)[:, 1][0]

    st.subheader(f"Default Probability: {probability:.2%}")

    if probability > 0.6:
        st.error("🔴 High Risk Customer")
    elif probability > 0.4:
        st.warning("🟠 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")

    # ===============================
    # SHAP (Lazy Loaded + Cached)
    # ===============================

    with st.spinner("Generating explanation..."):

        import shap  # lazy import

        @st.cache_resource
        def get_explainer(model):
            xgb_model = model.named_steps["classifier"]
            return shap.TreeExplainer(xgb_model)

        explainer = get_explainer(model)

        xgb_model = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        input_transformed = preprocessor.transform(input_data)

        shap_values = explainer.shap_values(input_transformed)

        st.subheader("🔎 Model Explanation")

        fig = plt.figure()

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_transformed[0]
            ),
            max_display=7
        )

        st.pyplot(fig)
