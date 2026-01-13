import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from pathlib import Path

# ------------------ PATH SETUP ------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "rf_placement_model.joblib"
FEATURE_PATH = BASE_DIR / "model" / "feature_columns.json"

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols

model, feature_cols = load_model()

# ------------------ APP SETUP ------------------
st.set_page_config(
    page_title="Engineering College Placement Prediction System",
    layout="wide"
)

st.title("🎓 Engineering College Placement Prediction System")
st.sidebar.success("Model loaded successfully")

# ------------------ SIDEBAR ------------------
page = st.sidebar.radio(
    "Select User Role:",
    ["Student Prediction", "T&P Dashboard"]
)

# ================== STUDENT SIDE ==================
if page == "Student Prediction":
    st.header("📋 Student Placement Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenth = st.number_input("10th Percentage", 0.0, 100.0, 85.0)
        twelfth = st.number_input("12th Percentage", 0.0, 100.0, 80.0)
        cgpa = st.number_input("BTech CGPA", 0.0, 10.0, 8.0)
        projects = st.number_input("Number of Projects", 0, 10, 2)
        internships = st.number_input("Internships Done", 0, 5, 0)

    with col2:
        tech_skills = st.number_input("Technical Skills Count", 0, 10, 5)
        soft_skills = st.number_input("Soft Skills Rating (1–10)", 1, 10, 7)
        aptitude = st.number_input("Aptitude Score (1–10)", 1, 10, 6)
        backlogs = st.number_input("Number of Backlogs", 0, 10, 0)

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"])

    if st.button("Predict Placement"):
        # Student input dictionary
        student = {
            "Gender": gender,
            "Branch": branch,
            "10th_Percentage": tenth,
            "12th_Percentage": twelfth,
            "BTech_CGPA": cgpa,
            "No_of_Projects": projects,
            "Internships": internships,
            "Technical_Skills_Count": tech_skills,
            "Soft_Skills_Rating": soft_skills,
            "Backlogs": backlogs,
            "Aptitude_Score": aptitude
        }

        # Convert to DataFrame
        sdf = pd.DataFrame([student])
        sdf = pd.get_dummies(sdf, drop_first=True)

        # Align with training features
        for col in feature_cols:
            if col not in sdf.columns:
                sdf[col] = 0
        sdf = sdf[feature_cols]

        # Prediction
        prob = model.predict_proba(sdf)[0][1]
        pred = model.predict(sdf)[0]

        if pred == 1:
            st.success(f"✅ Likely to be Placed (Probability: {prob*100:.2f}%)")
        else:
            st.error(f"❌ Not Likely to be Placed (Probability: {prob*100:.2f}%)")

        # ------------------ RADAR VISUAL ------------------
        radar_data = pd.DataFrame({
            "Category": ["10th", "12th", "CGPA", "Projects", "Technical", "Soft", "Aptitude"],
            "Score": [
                tenth / 10,
                twelfth / 10,
                cgpa,
                projects,
                tech_skills,
                soft_skills,
                aptitude
            ]
        })

        fig = px.line_polar(
            radar_data,
            r="Score",
            theta="Category",
            line_close=True,
            title="📈 Student Performance Profile"
        )
        st.plotly_chart(fig, use_container_width=True)

# ================== T&P DASHBOARD ==================
elif page == "T&P Dashboard":
    st.header("📊 Training & Placement Analytics Dashboard")

    uploaded = st.file_uploader(
        "Upload student dataset (CSV)",
        type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Metrics
        if "Placement" in df.columns:
            placed = df[df["Placement"] == 1].shape[0]
            not_placed = df[df["Placement"] == 0].shape[0]
            total = df.shape[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Students", total)
            c2.metric("Placed", placed)
            c3.metric("Not Placed", not_placed)

            # Placement Ratio
            fig1 = px.pie(
                df,
                names="Placement",
                title="Placement Ratio"
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Placement by Branch
        if {"Branch", "Placement"}.issubset(df.columns):
            fig2 = px.histogram(
                df,
                x="Branch",
                color="Placement",
                barmode="group",
                title="Placement by Branch"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # CGPA vs Placement
        if {"BTech_CGPA", "Placement"}.issubset(df.columns):
            fig3 = px.box(
                df,
                x="Placement",
                y="BTech_CGPA",
                title="CGPA Distribution vs Placement"
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Projects vs Placement
        if {"No_of_Projects", "Placement"}.issubset(df.columns):
            avg_proj = (
                df.groupby("Placement")["No_of_Projects"]
                .mean()
                .reset_index()
            )
            fig4 = px.bar(
                avg_proj,
                x="Placement",
                y="No_of_Projects",
                title="Average Projects vs Placement"
            )
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("📂 Please upload a CSV file to view placement analytics.")
