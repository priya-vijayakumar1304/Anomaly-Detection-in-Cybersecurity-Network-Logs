import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Cybersecurity Anomaly Detection", layout="wide")

st.title("🔐 Cybersecurity Anomaly Detection Dashboard")
st.write(
    "Upload network traffic logs to identify suspicious or anomalous activity."
)

# upload file
uploaded_file = st.file_uploader(
    "📁 Upload Network Log CSV", type=["csv"]
)

if uploaded_file:
    st.success("File uploaded successfully.")

    if st.button("🔍 Detect Anomalies"):
        with st.spinner("Analyzing network traffic..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv",
                )
            }

            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error("Error from prediction API.")
        else:
            result = response.json()

            total = result["total_records"]
            anomalies = result["anomalies_detected"]
            normals = result["normal_records"]
            severity = round((anomalies / total) * 100, 2)

            # metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", total)
            col2.metric("Anomalies Detected", anomalies)
            col3.metric("Normal Records", normals)
            col4.metric("Severity (%)", severity)

            # severity interpretation
            st.subheader("🚦 Network Risk Assessment")

            # Progress bar (0–100)
            st.progress(min(int(severity), 100))

            # Interpretation
            if severity < 10:
                st.success("🟢 Low Risk: Network traffic appears normal.")
            elif severity < 30:
                st.warning("🟠 Medium Risk: Suspicious activity detected.")
            else:
                st.error("🔴 High Risk: Immediate investigation recommended.")


            # visual insights
            st.subheader("📊 Anomaly Distribution")

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["Normal", "Anomaly"], [normals, anomalies])
            ax.set_ylabel("Records")
            ax.set_title("Anomaly Distribution")
            st.pyplot(fig, use_container_width=False)

            # table
            df = pd.DataFrame(result["data"])

            st.subheader("🚨 Detected Anomalies (Preview)")
            anomalies_df = df[df["anomaly_label"] == "Anomaly"]

            st.dataframe(anomalies_df.head(100), use_container_width=True)

            # feature level summary for anomalies
            st.subheader("📊 Anomalous Traffic Characteristics")

            key_features = [
                "dur",
                "spkts",
                "dpkts",
                "sbytes",
                "dbytes"
            ]

            # Keep only features that actually exist (safety)
            available_features = [f for f in key_features if f in anomalies_df.columns]

            if available_features:
                summary = anomalies_df[available_features].describe().loc[
                    ["mean", "max"]
                ]
                st.dataframe(summary, use_container_width=True)
            else:
                st.info("Feature summary not available for this dataset.")


            # download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="⬇️ Download Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="anomaly_detection_results.csv",
                mime="text/csv",
            )
