#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# LOAD MODEL PACKAGE
# -------------------------------
package = joblib.load('model_with_explainer.pkl')
model = package['model']
explainer = package['explainer']

# Ensure feature names are strings
features = [str(col) for col in package['features']]

# -------------------------------
# UI CONFIG
# -------------------------------
st.title("🏦 Smart Credit Risk Evaluator")

job_options = {
    "Unemployed/Unskilled Non-resident": 0,
    "Unskilled Resident": 1,
    "Skilled Employee": 2,
    "Highly Qualified/Management": 3
}

# -------------------------------
# USER INPUTS
# -------------------------------
age = st.number_input("Customer Age", 18, 100, 30)
amount = st.number_input("Loan Amount ($)", 100, 20000, 1000)
duration = st.slider("Loan Duration (Months)", 1, 72, 24)
selection = st.selectbox("Job Category", options=list(job_options.keys()))
job_value = job_options[selection]
checking = st.checkbox("Checking Account Status Unknown?")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if st.button("Analyze Risk"):

    # Create input dataframe with all features
    input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    input_df.columns = input_df.columns.astype(str)

    # Fill values
    input_df['Age'] = age
    input_df['Credit amount'] = amount
    input_df['Duration'] = duration
    input_df['Job'] = job_value
    input_df['Checking account_unknown'] = 1 if checking else 0

    # -------------------------------
    # PREDICTION
    # -------------------------------
    prob = model.predict_proba(input_df)[:, 0][0]

    # -------------------------------
    # HIGH RISK CASE
    # -------------------------------
    if prob >= 0.4:
        st.error(f"### ❌ REJECTED: High Risk (Score: {prob:.2f})")
        st.subheader("Top Factors Increasing Your Risk:")

        try:
            # -------------------------------
            # SHAP VALUES
            # -------------------------------
            shap_values = explainer.shap_values(input_df)

            # Handle all SHAP formats
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]

            # If still 2D → pick class column
            if len(np.array(shap_vals).shape) == 2:
                shap_vals = shap_vals[:, 1]

            shap_vals = np.array(shap_vals).flatten()

            # Final validation
            if len(shap_vals) != len(features):
                st.warning("⚠️ SHAP output mismatch. Skipping explanation.")
            else:
                risk_impact = pd.Series(shap_vals, index=features)

                # Top 3 risk drivers
                top_reasons = risk_impact.sort_values(ascending=False).head(3)

                for feature, impact_value in top_reasons.items():
                    if impact_value > 0:
                        st.write(f"⚠️ **{feature}** is increasing risk")

        except Exception as e:
            st.warning(f"SHAP explanation failed: {str(e)}")
 
        st.info("💡 Tip: Reduce loan amount or duration to improve approval chances.")

    # -------------------------------
    # LOW RISK CASE
    # -------------------------------
    else:
        st.success(f"✅ APPROVED: Low Risk (Score: {prob:.2f})")
        st.write(f"Probability of Default: **{prob:.2%}**")
        st.info("Recommendation: Approve application.")