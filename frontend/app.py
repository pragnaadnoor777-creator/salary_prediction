import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salary Prediction System",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ---------------- SESSION STATE ----------------
if "predicted_salary" not in st.session_state:
    st.session_state.predicted_salary = None

# ---------------- TITLE ----------------
st.title(" Salary Prediction System")

# =====================================================
# INPUT SECTION
# =====================================================

st.header(" Enter Employee Details")

exp = st.text_input("Experience (years)")
edu = st.selectbox("Education Level",['Bachelor','Master','PhD'])
job = st.selectbox("Job Title",['Engineer','Executive','Intern','Analyst','Manager',''])

# =====================================================
# SALARY PREDICTION
# =====================================================

if st.button("Predict Salary"):
    try:
        response = requests.get(
            "http://127.0.0.1:8000/pred_sal",
            params={"exp": exp, "edu": edu, "job": job},
            timeout=10
        )

        data = response.json()

        st.session_state.predicted_salary = data["predicted_salary"]

        st.success(f"Predicted Salary: ₹ {st.session_state.predicted_salary:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =====================================================
# DATA VISUALIZATION
# =====================================================

st.write("---")
st.header("Data Visualization")

if st.button("Show Salary vs Experience Graph"):
    try:
        params = {
            "exp": exp,
            "edu": edu,
            "job": job
        }

        res = requests.get(
            "http://127.0.0.1:8000/view_graph",
            params=params,
            timeout=30
        )

        if res.status_code == 200:
            st.image(res.content, caption="Salary vs Experience")
        else:
            st.error("Could not load graph")

    except Exception as e:
        st.error(f"Could not load graph: {e}")

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.write("---")
st.header(" Model Performance")

if st.button("Check Model Metrics"):
    try:
        acc_response = requests.get(
            f"{API_URL}/accuracy",
            timeout=10
        )

        metrics = acc_response.json()

        metrics_df = pd.DataFrame(
            metrics.items(),
            columns=["Metric", "Value"]
        )

        st.table(metrics_df)

    except Exception as e:
        st.error(f"Could not load metrics: {e}")

#===============================================
#        SHAP WATERFALL
#===============================================

st.write("---")
st.header("SHAP Waterfall Explanation")
if st.button("Show SHAP Waterfall"):
    if st.session_state.predicted_salary is None:
        st.warning("Please predict salary first.")
    else:
        try:
            res = requests.get(
                "http://127.0.0.1:8000/shap_waterfall",
                params={
                    "exp": exp,
                    "edu": edu,
                    "job": job
                },
                timeout=60
            )

            if res.status_code == 200:
                st.image(res.content, caption="SHAP Waterfall Plot")
            else:
                st.error("Could not load SHAP waterfall")

        except Exception as e:
            st.error(f"SHAP error: {e}")

