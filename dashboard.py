import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import shap
import torch

st.set_page_config(page_title="Solar Forecasting Dashboard", layout="wide")

st.title("⚡ Solar Forecasting Dashboard (HGAT+GRU)")
st.markdown("Professional interface with predictions, metrics, visualization, and explainability.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Single Prediction", "Batch Prediction", "Explainability"])

# --- Global features list (default zeros) ---
features = [0.0] * 10

# --- Single Prediction Page ---
if page == "Single Prediction":
    st.header("🔹 Single Prediction")
    cols = st.columns(5)
    for i in range(10):
        with cols[i % 5]:
            val = st.number_input(f"Feature {i+1}", value=0.0, key=f"f{i}")
            features[i] = val

    if st.button("Predict AC_POWER (Single Row)"):
        response = requests.post("http://127.0.0.1:8000/predict", json={"features": features})
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Predicted AC_POWER: {result['prediction']:.2f}")
        else:
            st.error("❌ Error: Could not get prediction from backend")

# --- Batch Prediction Page ---
elif page == "Batch Prediction":
    st.header("🔹 Batch Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Upload CSV with 10 feature columns + optional Actual_AC_POWER", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Auto-rename columns if missing headers
        if list(df.columns) == list(range(10)):
            df.columns = [f"Feature{i+1}" for i in range(10)]

        st.write("📊 Uploaded Data Preview:", df.head())

        predictions = []
        for _, row in df.iloc[:, :10].iterrows():  # only first 10 columns are features
            row_features = row.tolist()
            response = requests.post("http://127.0.0.1:8000/predict", json={"features": row_features})
            if response.status_code == 200:
                result = response.json()
                predictions.append(result['prediction'])
            else:
                predictions.append(None)

        df["Predicted_AC_POWER"] = predictions
        st.write("📊 Results with Predictions:", df)

        # --- Metrics if Actual_AC_POWER exists ---
        if "Actual_AC_POWER" in df.columns:
            mae = mean_absolute_error(df["Actual_AC_POWER"], df["Predicted_AC_POWER"])
            rmse = np.sqrt(mean_squared_error(df["Actual_AC_POWER"], df["Predicted_AC_POWER"]))
            r2 = r2_score(df["Actual_AC_POWER"], df["Predicted_AC_POWER"])

            st.subheader("📐 Model Performance Metrics")
            st.write(f"- MAE: {mae:.2f}")
            st.write(f"- RMSE: {rmse:.2f}")
            st.write(f"- R² Score: {r2:.2f}")

            # --- Visualization ---
            st.subheader("📈 Predicted vs Actual Plot")
            fig, ax = plt.subplots()
            ax.plot(df.index, df["Actual_AC_POWER"], marker="o", label="Actual AC_POWER")
            ax.plot(df.index, df["Predicted_AC_POWER"], marker="x", label="Predicted AC_POWER")
            ax.set_xlabel("Row Index")
            ax.set_ylabel("AC_POWER")
            ax.set_title("Predicted vs Actual AC_POWER")
            ax.legend()
            st.pyplot(fig)

        # Download option
        st.download_button("⬇️ Download Predictions CSV", df.to_csv(index=False), "predictions.csv")

# --- Explainability Page ---
elif page == "Explainability":
    st.header("🔹 Model Explainability (SHAP)")

    if st.button("Generate SHAP Feature Importance"):
        # Example: take one row of features (first row of uploaded CSV if available)
        if 'uploaded_file' in locals() and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            sample_features = df.iloc[0, :10].tolist()
        else:
            sample_features = features  # use single input box values (default zeros if not set)

        # Convert to numpy
        x = np.array([sample_features], dtype=np.float32)

        # Load model again (same as backend)
        class HGAT_GRU(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(HGAT_GRU, self).__init__()
                self.attn = torch.nn.Linear(input_dim, hidden_dim)
                self.dropout = torch.nn.Dropout(0.2)
                self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
                self.fc = torch.nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                h = torch.relu(self.attn(x))
                h = self.dropout(h)
                h = h.unsqueeze(1)
                h, _ = self.gru(h)
                h = h.squeeze(1)
                out = self.fc(h)
                return out

        model = HGAT_GRU(input_dim=10, hidden_dim=32, output_dim=1)
        model.load_state_dict(torch.load("models/hgat_gru_model.pth"))
        model.eval()

        # Define a wrapper function for SHAP
        def model_predict(data_as_numpy):
            data_tensor = torch.tensor(data_as_numpy, dtype=torch.float32)
            with torch.no_grad():
                preds = model(data_tensor).numpy()
            return preds

        # Background dataset for SHAP
        background = np.random.rand(50, 10).astype(np.float32)

        # SHAP explainability
        explainer = shap.Explainer(model_predict, background)
        shap_values = explainer(x)

        # --- Bar Plot ---
        st.subheader("📊 SHAP Feature Importance (Bar Plot)")
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())

        # --- Force Plot ---
        st.subheader("📊 SHAP Force Plot (Red/Blue Contributions)")
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        st.pyplot(plt.gcf())
