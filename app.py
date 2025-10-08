import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from math import ceil

# -----------------------------
# Load model + encoder
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/food_forecast_model.joblib")
    le = joblib.load("models/label_encoder.joblib")
    return model, le

model, le = load_model()

# -----------------------------
# App layout
# -----------------------------
st.set_page_config(page_title="Cafeteria Food Forecast", layout="wide")
st.title("🍱 Cafeteria Food Demand Forecast")
st.write("Predict tomorrow’s food requirements to minimize waste and shortages.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header(" Configuration")
safety = st.sidebar.slider("Safety Buffer (%)", 0, 50, 10)
show_charts = st.sidebar.checkbox("Show visualizations", True)

# -----------------------------
# Simulated / uploaded data
# -----------------------------
uploaded = st.file_uploader("Upload recent cafeteria sales (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(" Data uploaded successfully!")
else:
    df = pd.read_csv("data/cafeteria_sales.csv")
    st.info("Using sample cafeteria data.")

# -----------------------------
# Feature extraction
# -----------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["item_name", "date"])
last = df.groupby("item_name").last().reset_index()

# Safe fallback if features missing
for col in ["lag_1", "lag_7", "roll7", "temperature", "is_holiday", "day_of_week", "month"]:
    if col not in last.columns:
        last[col] = np.random.randint(0, 10, len(last))

# Encode item names
last["item_id_enc"] = le.transform(last["item_name"])

# Predict
features = ["item_id_enc","lag_1","lag_7","roll7","temperature","is_holiday","day_of_week","month"]
preds = model.predict(last[features])
preds = np.maximum(0, np.round(preds))

# Add safety buffer
prep = np.ceil(preds * (1 + safety / 100)).astype(int)

# Output table
results = pd.DataFrame({
    "Item": last["item_name"],
    "Predicted Sales": preds.astype(int),
    f"Suggested Prep (+{safety}% buffer)": prep
}).sort_values(by="Predicted Sales", ascending=False)

# -----------------------------
# Display results
# -----------------------------
st.subheader(" Tomorrow’s Forecast")
st.dataframe(results, use_container_width=True)

# -----------------------------
# Optional: Visualization
# -----------------------------
if show_charts:
    st.markdown("### 🍽️ Sales Prediction Overview")
    fig = px.bar(
        results,
        x="Item",
        y="Predicted Sales",
        text="Predicted Sales",
        color="Item",
        title="Predicted Demand per Item"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed for Hackathon 2025 — Smart Cafeteria Initiative ")
