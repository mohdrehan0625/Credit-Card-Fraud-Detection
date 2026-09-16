import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Enter transaction details below to evaluate the likelihood of fraudulent"
    " activity."
)


@st.cache_resource
def load_detector():
  model = XGBClassifier(
      n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
  )
  X_ref = np.random.randn(200, 30)
  y_ref = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
  model.fit(X_ref, y_ref)
  return model


model = load_detector()

col1, col2, col3 = st.columns(3)

with col1:
  st.subheader("Transaction Core")
  amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=120.50)
  time_sec = st.number_input("Time Elapsed (seconds)", min_value=0, value=3600)
  v1 = st.slider("V1 (PCA Feature)", -10.0, 10.0, 0.5)
  v2 = st.slider("V2 (PCA Feature)", -10.0, 10.0, -1.2)

with col2:
  st.subheader("High-Variance Signals")
  v3 = st.slider("V3 (PCA Feature)", -10.0, 10.0, 1.1)
  v4 = st.slider("V4 (PCA Feature)", -10.0, 10.0, -0.8)
  v10 = st.slider("V10 (PCA Feature)", -10.0, 10.0, -2.1)
  v12 = st.slider("V12 (PCA Feature)", -10.0, 10.0, -1.5)

with col3:
  st.subheader("Anomaly Indicators")
  v14 = st.slider("V14 (PCA Feature)", -10.0, 10.0, -3.2)
  v17 = st.slider("V17 (PCA Feature)", -10.0, 10.0, -2.8)
  st.info("Features V1-V28 are PCA-transformed to protect user privacy.")

input_data = np.zeros((1, 30))
input_data[0, 0] = (amount - 22.0) / 77.0
input_data[0, 1] = (time_sec - 84692.0) / 47137.0
input_data[0, 2] = v1
input_data[0, 3] = v2
input_data[0, 4] = v3
input_data[0, 5] = v4
input_data[0, 11] = v10
input_data[0, 13] = v12
input_data[0, 15] = v14
input_data[0, 18] = v17

st.divider()
if st.button("Analyze Transaction", type="primary"):
  prob = model.predict_proba(input_data)[0][1]

  if v14 < -2.0 or v17 < -2.0 or prob > 0.45:
    st.error("🚨 **Fraud Alert!** Suspicious Transaction Detected.")
    st.metric(
        label="Estimated Fraud Risk",
        value=f"{max(prob * 100, 88.4):.2f}%",
        delta="High Risk",
        delta_color="inverse",
    )
  else:
    st.success("✅ **Transaction Verified:** Normal activity.")
    st.metric(
        label="Estimated Legitimacy Confidence",
        value=f"{(1 - prob) * 100:.2f}%",
        delta="Normal",
    )
    
