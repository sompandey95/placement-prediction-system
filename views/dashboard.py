import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import load_comparison_results
from src.predict import load_artifacts, predict_student


model, feature_cols = load_artifacts()


def show():
    try:
        st.header("📊 Training & Placement Analytics Dashboard")
        tab1, tab2 = st.tabs(["📁 Upload & Analyze", "🤖 Batch Prediction"])

        with tab1:
            st.subheader("Upload Student Dataset for Analytics")
            uploaded = st.file_uploader("Upload student dataset (CSV)", type=["csv"])

            if uploaded is None:
                st.info("📂 Upload a CSV file containing student data to view placement analytics.")
            else:
                df = pd.read_csv(uploaded)
                st.success(f"✅ Dataset loaded: {df.shape[0]} students, {df.shape[1]} columns")

                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)

                if "Placement_Status" in df.columns:
                    placed = df[df["Placement_Status"] == 1].shape[0]
                    not_placed = df[df["Placement_Status"] == 0].shape[0]
                    placement_rate = placed / df.shape[0] * 100

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Students", df.shape[0])
                    c2.metric("Placed", placed)
                    c3.metric("Not Placed", not_placed)
                    c4.metric("Placement Rate", f"{placement_rate:.1f}%")

                    left_col, right_col = st.columns(2)
                    with left_col:
                        pie_fig = px.pie(
                            df,
                            names="Placement_Status",
                            color="Placement_Status",
                            title="Placement Ratio",
                            color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                        )
                        st.plotly_chart(pie_fig, width="stretch")

                    with right_col:
                        if "Branch" in df.columns:
                            branch_fig = px.histogram(
                                df,
                                x="Branch",
                                color="Placement_Status",
                                barmode="group",
                                title="Placement by Branch",
                            )
                            st.plotly_chart(branch_fig, width="stretch")

                if "BTech_CGPA" in df.columns and "Placement_Status" in df.columns:
                    left_col, right_col = st.columns(2)
                    with left_col:
                        cgpa_fig = px.box(
                            df,
                            x="Placement_Status",
                            y="BTech_CGPA",
                            color="Placement_Status",
                            title="CGPA Distribution by Placement",
                            color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                        )
                        st.plotly_chart(cgpa_fig, width="stretch")

                    with right_col:
                        if "No_of_Projects" in df.columns:
                            projects_fig = px.box(
                                df,
                                x="Placement_Status",
                                y="No_of_Projects",
                                color="Placement_Status",
                                title="Projects Distribution by Placement",
                                color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                            )
                            st.plotly_chart(projects_fig, width="stretch")

                numeric_df = df.select_dtypes(include="number")
                if len(numeric_df.columns) >= 5:
                    st.subheader("🔥 Feature Correlation Heatmap")
                    corr = numeric_df.corr().round(2)
                    heatmap_fig = px.imshow(
                        corr,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Feature Correlation Matrix",
                        aspect="auto",
                    )
                    st.plotly_chart(heatmap_fig, width="stretch")

                has_identifier = "Name" in df.columns or "Student_ID" in df.columns
                if "Placement_Status" in df.columns and has_identifier:
                    st.subheader("⚠️ Students at Risk (Not Placed)")
                    at_risk = df[df["Placement_Status"] == 0]
                    risk_columns = [
                        col
                        for col in [
                            "Student_ID",
                            "Name",
                            "Branch",
                            "BTech_CGPA",
                            "Backlogs",
                            "No_of_Projects",
                        ]
                        if col in at_risk.columns
                    ]
                    st.dataframe(at_risk[risk_columns], use_container_width=True)
                    st.warning(f"{len(at_risk)} students predicted as Not Placed")

        with tab2:
            st.subheader("🤖 Batch Placement Prediction")
            st.info(
                "Upload a CSV with student details. The model will predict placement for each student and provide a downloadable result."
            )

            with st.expander("📋 Expected CSV Columns"):
                st.code(
                    "Gender, Branch, 10th_Percentage, 12th_Percentage, BTech_CGPA, No_of_Projects, Internships, Technical_Skills_Count, Soft_Skills_Rating, Backlogs, Aptitude_Score"
                )

            batch_file = st.file_uploader(
                "Upload student CSV for batch prediction",
                type=["csv"],
                key="batch",
            )

            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                st.subheader("Batch Dataset Preview")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button("🚀 Run Batch Prediction"):
                    expected_columns = [
                        "Gender",
                        "Branch",
                        "10th_Percentage",
                        "12th_Percentage",
                        "BTech_CGPA",
                        "No_of_Projects",
                        "Internships",
                        "Technical_Skills_Count",
                        "Soft_Skills_Rating",
                        "Backlogs",
                        "Aptitude_Score",
                    ]
                    missing_columns = [
                        col for col in expected_columns if col not in batch_df.columns
                    ]

                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        probs = []
                        preds = []
                        progress = st.progress(0)
                        status = st.empty()

                        for i, row in batch_df.iterrows():
                            student_dict = {
                                "Gender": row["Gender"],
                                "Branch": row["Branch"],
                                "10th_Percentage": row["10th_Percentage"],
                                "12th_Percentage": row["12th_Percentage"],
                                "BTech_CGPA": row["BTech_CGPA"],
                                "No_of_Projects": row["No_of_Projects"],
                                "Internships": row["Internships"],
                                "Technical_Skills_Count": row["Technical_Skills_Count"],
                                "Soft_Skills_Rating": row["Soft_Skills_Rating"],
                                "Backlogs": row["Backlogs"],
                                "Aptitude_Score": row["Aptitude_Score"],
                            }
                            prob, pred = predict_student(student_dict, model, feature_cols)
                            probs.append(prob)
                            preds.append(pred)
                            progress.progress((i + 1) / len(batch_df))
                            status.text(
                                f"Processing student {i + 1} of {len(batch_df)}..."
                            )

                        status.text("✅ Prediction complete!")
                        batch_df["Placement_Prediction"] = preds
                        batch_df["Placement_Probability_%"] = [
                            round(p * 100, 2) for p in probs
                        ]
                        batch_df["Result"] = batch_df["Placement_Prediction"].map(
                            {1: "Placed ✅", 0: "Not Placed ❌"}
                        )

                        total = len(batch_df)
                        predicted_placed = int(sum(preds))
                        predicted_not_placed = total - predicted_placed

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Students", total)
                        c2.metric("Predicted Placed", predicted_placed)
                        c3.metric("Predicted Not Placed", predicted_not_placed)

                        st.subheader("Batch Prediction Results")
                        st.dataframe(batch_df, use_container_width=True)

                        csv_data = batch_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_data,
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                        )

    except Exception as e:
        import traceback

        st.error(f"Dashboard error: {e}")
        st.code(traceback.format_exc())
