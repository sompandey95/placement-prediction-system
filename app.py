import streamlit as st
from views import student, dashboard

st.set_page_config(
    page_title="AI-Powered Student Placement Intelligence Platform",
    layout="wide"
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] ~ div .block-container {
        padding-top: 0.5rem !important;
        overflow: visible !important;
    }
    div[data-testid="stAppViewContainer"] > section > div {
        overflow: visible !important;
    }
    .app-main-title {
        font-size:50px;
        font-weight: 700;
        margin: 20px 0 6px 0 !important;
        padding: 0;
        line-height: 1.3;
        white-space: normal;
        word-wrap: break-word;
        color:white;
        display: block;
    }

    /* App-level floating chatbot overlay */
    [data-testid="stCustomComponentV1"] {
        position: fixed !important;
        top: auto !important;
        left: auto !important;
        right: 24px !important;
        bottom: 24px !important;
        width: 420px !important;
        height: 640px !important;
        z-index: 2147483640 !important;
        pointer-events: none !important;
        overflow: visible !important;
        border: none !important;
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stCustomComponentV1"] > div {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
        overflow: visible !important;
    }

    [data-testid="stCustomComponentV1"] iframe {
        position: fixed !important;
        top: auto !important;
        left: auto !important;
        right: 24px !important;
        bottom: 24px !important;
        width: 420px !important;
        height: 640px !important;
        max-width: calc(100vw - 32px) !important;
        max-height: calc(100vh - 32px) !important;
        z-index: 2147483647 !important;
        pointer-events: auto !important;
        overflow: visible !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    div:has(> [data-testid="stCustomComponentV1"]),
    div:has(> div > [data-testid="stCustomComponentV1"]) {
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: visible !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="app-main-title">🎓 AI-Powered Student Placement Intelligence Platform</h1>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Select User Role:",
    ["Student Prediction", "T&P Dashboard"]
)

if page == "Student Prediction":
    student.show()
elif page == "T&P Dashboard":
    dashboard.show()

if page == "Student Prediction" and "latest_prediction" in st.session_state:
    prediction_context = st.session_state["latest_prediction"]
    student.render_floating_chat(
        student_dict=prediction_context["student_dict"],
        prob=prediction_context["prob"],
        skills=st.session_state.get("student_skills", []),
        pred=prediction_context["pred"],
    )
