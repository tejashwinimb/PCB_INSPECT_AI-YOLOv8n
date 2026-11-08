import streamlit as st
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from PIL import Image
import os
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PCB INSPECT AI",
    page_icon="🔬",
    layout="wide"
)

# ---------------- DARK THEME ----------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #38bdf8;
}
[data-testid="stMetric"] {
    background: #172338;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #26364d;
}
.stButton>button {
    background: #0891b2;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.title("🔬 PCB INSPECT AI")
st.markdown("### AI-Powered PCB Defect Detection System")
st.caption("YOLOv8n | Deep Learning | Automated Inspection")

st.divider()

# ---------------- MODEL ----------------

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    if not os.path.exists(MODEL_PATH):
        st.error("best.pt not found. Keep it in the app.py folder.")
        st.stop()

    model = load_model()
    model_loaded = True

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Inspection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.10, 1.00, 0.25, 0.05
)

st.sidebar.success("YOLOv8n Model Loaded")

# ---------------- METRICS ----------------

st.subheader("📊 Model Information")

c1, c2, c3 = st.columns(3)

c1.metric("Model", "YOLOv8n")
c2.metric("Classes", len(model.names))
c3.metric("Parameters", "From trained model")

st.divider()

# ---------------- IMAGE UPLOAD ----------------

st.subheader("🖼️ PCB Image Inspection")

uploaded_file = st.file_uploader(
    "Upload a PCB image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original PCB Image", width="stretch")

    if st.button("🚀 START PCB INSPECTION", use_container_width=True):

        with st.spinner("AI is inspecting the PCB..."):

            start_time = time.perf_counter()

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            elapsed = time.perf_counter() - start_time

            result = results[0]

            annotated = result.plot()
            annotated = annotated[:, :, ::-1]

            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    score = float(box.conf.item())

                    detections.append({
                        "Defect": model.names[class_id],
                        "Confidence": round(score * 100, 2)
                    })

        st.success("Inspection completed!")

        st.subheader("🔍 Detection Result")

        left, right = st.columns(2)

        with left:
            st.image(image, caption="Original Image", width="stretch")

        with right:
            st.image(
                annotated,
                caption="AI Detection Output",
                width="stretch"
            )

        total = len(detections)

        st.divider()
        st.subheader("📈 Inspection Summary")

        a, b, c = st.columns(3)

        a.metric("Total Defects", total)
        b.metric("Inference Time", f"{elapsed:.3f} sec")
        c.metric(
            "Board Status",
            "DEFECTS FOUND" if total else "NO DEFECTS DETECTED"
        )

        if detections:

            df = pd.DataFrame(detections)

            st.subheader("Defect Detection Table")
            st.dataframe(df, use_container_width=True)

            st.subheader("Defect Distribution")

            counts = df["Defect"].value_counts().reset_index()
            counts.columns = ["Defect", "Count"]

            fig = px.bar(
                counts,
                x="Defect",
                y="Count",
                color="Defect",
                template="plotly_dark",
                title="Detected Defects by Class"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "📥 Download Inspection Report",
                df.to_csv(index=False),
                file_name="pcb_inspection_report.csv",
                mime="text/csv"
            )

        else:
            st.info("No defects detected at the selected confidence threshold.")

        st.caption(
            "Inference time measures model prediction on this image; "
            "it does not include image upload or dashboard rendering."
        )

# ---------------- TRAINING RESULTS ----------------

st.divider()
st.subheader("📊 YOLOv8n Training Performance")

if os.path.exists("results.csv"):

    training = pd.read_csv("results.csv")
    training.columns = training.columns.str.strip()

    st.write("Actual training metrics from your results.csv")

    st.dataframe(training, use_container_width=True)

    metric_columns = [
        col for col in training.columns
        if any(x in col.lower() for x in [
            "precision", "recall", "map50", "map50-95"
        ])
    ]

    if metric_columns:

        fig = px.line(
            training,
            y=metric_columns,
            title="Training Performance Across Epochs",
            markers=True,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("results.csv not found.")

st.divider()

st.caption("PCB INSPECT AI | YOLOv8n-Based PCB Defect Inspection")