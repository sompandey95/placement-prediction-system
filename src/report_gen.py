import datetime
import logging
from pathlib import Path

from fpdf import FPDF


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_FEATURE_LABELS = {
    "BTech_CGPA": "B.Tech CGPA",
    "10th_Percentage": "10th Percentage",
    "12th_Percentage": "12th Percentage",
    "No_of_Projects": "No. of Projects",
    "Technical_Skills_Count": "Technical Skills",
    "Soft_Skills_Rating": "Soft Skills Rating",
    "Aptitude_Score": "Aptitude Score",
    "Backlogs": "Backlogs",
    "Internships": "Internships",
    "Gender_Male": "Gender: Male",
    "Branch_CSE": "Branch: CSE",
    "Branch_IT": "Branch: IT",
    "Branch_ECE": "Branch: ECE",
    "Branch_EEE": "Branch: EEE",
    "Branch_ME": "Branch: ME",
    "Branch_CIVIL": "Branch: Civil",
}


def _latin1_safe(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


class PlacementReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(
            0,
            12,
            _latin1_safe("AI-Powered Student Placement Intelligence Platform"),
            border=0,
            ln=True,
            align="C",
        )
        self.set_font("Arial", "", 11)
        self.cell(
            0,
            8,
            _latin1_safe("Student Placement Report"),
            border=0,
            ln=True,
            align="C",
        )
        self.line(10, 30, 200, 30)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(
            0,
            10,
            _latin1_safe(
                f"Page {self.page_no()} | Generated on {datetime.date.today()}"
            ),
            align="C",
        )


def generate_report(student_dict, prob, pred, shap_df):
    logger.info("Generating placement PDF report")
    pdf = PlacementReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, _latin1_safe("Prediction Result"), ln=True)
    pdf.set_font("Arial", "", 11)
    result_text = "LIKELY TO BE PLACED" if pred == 1 else "NOT LIKELY TO BE PLACED"
    probability_text = f"Placement Probability: {prob * 100:.1f}%"
    if pred == 1:
        pdf.set_fill_color(46, 204, 113)
    else:
        pdf.set_fill_color(231, 76, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, _latin1_safe(result_text), ln=True, fill=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, _latin1_safe(probability_text), ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, _latin1_safe("Student Academic Profile"), ln=True)
    pdf.set_font("Arial", "", 10)
    fields = [
        ("Gender", student_dict.get("Gender", "-")),
        ("Branch", student_dict.get("Branch", "-")),
        ("10th Percentage", f"{student_dict.get('10th_Percentage', 0):.1f}%"),
        ("12th Percentage", f"{student_dict.get('12th_Percentage', 0):.1f}%"),
        ("BTech CGPA", f"{student_dict.get('BTech_CGPA', 0):.2f}"),
        ("Number of Projects", str(student_dict.get("No_of_Projects", 0))),
        ("Internships", str(student_dict.get("Internships", 0))),
        ("Technical Skills", str(student_dict.get("Technical_Skills_Count", 0))),
        ("Soft Skills Rating", f"{student_dict.get('Soft_Skills_Rating', 0)}/10"),
        ("Aptitude Score", f"{student_dict.get('Aptitude_Score', 0)}/10"),
        ("Backlogs", str(student_dict.get("Backlogs", 0))),
    ]

    for row_index, i in enumerate(range(0, len(fields), 2)):
        fill = row_index % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)

        left_label, left_value = fields[i]
        right_label, right_value = fields[i + 1] if i + 1 < len(fields) else ("", "")

        pdf.set_font("Arial", "B", 10)
        pdf.cell(36, 8, _latin1_safe(f"{left_label}:"), fill=fill)
        pdf.set_font("Arial", "", 10)
        pdf.cell(54, 8, _latin1_safe(left_value), fill=fill)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(36, 8, _latin1_safe(f"{right_label}:") if right_label else "", fill=fill)
        pdf.set_font("Arial", "", 10)
        pdf.cell(54, 8, _latin1_safe(right_value), ln=True, fill=fill)

    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, _latin1_safe("Key Factors Affecting Prediction (SHAP Analysis)"), ln=True)
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(90, 8, _latin1_safe("Feature"), fill=True)
    pdf.cell(50, 8, _latin1_safe("SHAP Value"), fill=True)
    pdf.cell(50, 8, _latin1_safe("Impact"), ln=True, fill=True)

    if shap_df is None or shap_df.empty:
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(
            190,
            8,
            _latin1_safe("SHAP analysis unavailable for this prediction."),
            ln=True,
            fill=True,
        )
    else:
        for idx, (_, row) in enumerate(shap_df.iterrows()):
            fill = idx % 2 == 0
            if fill:
                pdf.set_fill_color(245, 245, 245)
            else:
                pdf.set_fill_color(255, 255, 255)

            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(0, 0, 0)
            raw_feature_name = str(row["Feature"])
            feature_name = REPORT_FEATURE_LABELS.get(
                raw_feature_name, raw_feature_name.replace("_", " ").title()
            )
            pdf.cell(90, 8, _latin1_safe(feature_name), fill=fill)
            pdf.cell(50, 8, _latin1_safe(f"{row['SHAP Value']:.4f}"), fill=fill)

            raw_impact = str(row.get("Impact", ""))
            if "Positive" in raw_impact or float(row.get("SHAP Value", 0)) > 0:
                impact_text = "Positive"
                pdf.set_fill_color(46, 204, 113)
            else:
                impact_text = "Negative"
                pdf.set_fill_color(231, 76, 60)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(50, 8, _latin1_safe(impact_text), ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)

    pdf.ln(4)

    recommendations = []
    if student_dict.get("Backlogs", 0) > 0:
        recommendations.append(
            "Clear your backlogs. Backlogs are among the strongest negative signals in placement."
        )
    if student_dict.get("No_of_Projects", 0) < 2:
        recommendations.append(
            "Work on at least 2-3 projects. Project count is a top feature in placement prediction."
        )
    if student_dict.get("BTech_CGPA", 0) < 7.0:
        recommendations.append(
            "Aim to improve your CGPA above 7.0. Academic performance significantly affects shortlisting."
        )
    if student_dict.get("Technical_Skills_Count", 0) < 5:
        recommendations.append(
            "Add more technical skills. Aim for at least 5 certifiable skills."
        )
    if student_dict.get("Internships", 0) == 0:
        recommendations.append(
            "Try to complete at least one internship before placement season."
        )
    if student_dict.get("Aptitude_Score", 0) < 6:
        recommendations.append(
            "Practice aptitude tests regularly. Many companies filter on aptitude scores."
        )
    if pred == 1 and student_dict.get("Backlogs", 0) == 0 and student_dict.get("No_of_Projects", 0) >= 2:
        recommendations.append(
            "You have a strong profile. Focus on interview preparation now."
        )

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, _latin1_safe("Personalized Recommendations"), ln=True)
    pdf.set_font("Arial", "", 10)
    if recommendations:
        for rec in recommendations:
            pdf.cell(0, 7, _latin1_safe(f"  * {rec}"), ln=True)
    else:
        pdf.cell(
            0,
            7,
            _latin1_safe("  * Strong profile! Focus on interview preparation."),
            ln=True,
        )

    return bytes(pdf.output())
