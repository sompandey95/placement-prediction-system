"""Generate a structured placement roadmap using Azure OpenAI chat completions."""

import json
import logging
import re

import streamlit as st
from src.advisor import chat_complete


logger = logging.getLogger(__name__)

FIELD_RESOURCES = {
    "AI/ML & Data Science": {
        "learning_path": ["Andrew Ng ML Course (Coursera)", "fast.ai Deep Learning", "Kaggle Learn"],
        "practice": ["Kaggle competitions", "UCI ML Repository datasets", "Papers with Code"],
        "companies": ["Google DeepMind", "Microsoft AI", "Amazon", "Flipkart Data Science", "Mu Sigma", "Fractal Analytics"],
        "certifications": ["Google ML Certificate", "AWS Machine Learning Specialty", "TensorFlow Developer Certificate"],
        "github_projects": ["Image classifier with deployment", "NLP sentiment analyzer", "End-to-end ML pipeline with MLflow"],
        "interview_prep": ["ML system design questions", "Statistics & probability", "SQL for data roles", "Coding in Python"],
    },
    "Web Development (Full Stack)": {
        "learning_path": ["The Odin Project", "Full Stack Open (Helsinki)", "Frontend Masters"],
        "practice": ["Build 3 full-stack projects", "Contribute to open source", "Deploy on Vercel/Render"],
        "companies": ["Razorpay", "Swiggy", "Zepto", "CRED", "Atlassian", "ThoughtWorks"],
        "certifications": ["Meta Frontend Developer (Coursera)", "AWS Cloud Practitioner"],
        "github_projects": ["E-commerce app with auth", "Real-time chat app", "Portfolio with CMS"],
        "interview_prep": ["JavaScript fundamentals", "React patterns", "System design basics", "REST API design"],
    },
    "Cloud & DevOps": {
        "learning_path": ["AWS Skill Builder", "KodeKloud for Kubernetes", "Linux Foundation courses"],
        "practice": ["Set up CI/CD pipeline on GitHub Actions", "Deploy on AWS free tier", "Kubernetes local cluster"],
        "companies": ["Accenture Cloud", "Infosys Cloud", "IBM", "Rackspace", "Wipro"],
        "certifications": ["AWS Solutions Architect Associate", "CKA (Kubernetes)", "Azure Fundamentals"],
        "github_projects": ["Dockerized microservice app", "Terraform infrastructure as code", "CI/CD pipeline project"],
        "interview_prep": ["Networking basics (TCP/IP, DNS)", "Linux commands", "Cloud pricing models", "Monitoring & logging"],
    },
    "Cybersecurity": {
        "learning_path": ["TryHackMe", "HackTheBox", "CompTIA Security+ prep"],
        "practice": ["CTF competitions", "Vulnerable VM labs (DVWA)", "Bug bounty basics"],
        "companies": ["Deloitte Cyber", "PwC Cybersecurity", "KPMG", "Quick Heal", "Paladion"],
        "certifications": ["CompTIA Security+", "CEH (Ethical Hacking)", "OSCP"],
        "github_projects": ["Network scanner tool", "Password strength analyzer", "Log analysis script"],
        "interview_prep": ["OWASP Top 10", "Network protocols", "Cryptography basics", "Incident response"],
    },
    "Software Engineering (General)": {
        "learning_path": ["NeetCode DSA roadmap", "CS50 (Harvard)", "System Design Primer (GitHub)"],
        "practice": ["LeetCode 150 problems", "Mock interviews on Pramp", "Build 2 full projects"],
        "companies": ["TCS", "Infosys", "Wipro", "Cognizant", "Capgemini", "HCL", "Tech Mahindra"],
        "certifications": ["Oracle Java Certified", "AWS Cloud Practitioner"],
        "github_projects": ["REST API with CRUD", "CLI tool in Python/Java", "Portfolio website"],
        "interview_prep": ["DSA basics", "OOP concepts", "SQL queries", "Aptitude & reasoning"],
    },
    "Mobile Development": {
        "learning_path": ["Android Developer (developer.android.com)", "Flutter docs", "iOS (100 Days of SwiftUI)"],
        "practice": ["Publish 1 app on Play Store/App Store", "Build offline-first app"],
        "companies": ["Zomato", "Ola", "PhonePe", "Meesho", "ShareChat", "InMobi"],
        "certifications": ["Google Associate Android Developer", "Flutter Certified Developer"],
        "github_projects": ["Task manager app", "News reader app with API", "Offline note-taking app"],
        "interview_prep": ["Activity/Fragment lifecycle", "State management patterns", "API integration", "App performance"],
    },
    "Data Engineering & Databases": {
        "learning_path": ["DataTalks.Club DE Zoomcamp", "Mode SQL Tutorial", "Spark with Python (Udemy)"],
        "practice": ["Build ETL pipelines", "Model a database schema", "Airflow DAGs"],
        "companies": ["Walmart Labs", "Juspay", "Urban Company", "Dunzo", "Nykaa"],
        "certifications": ["Google Professional Data Engineer", "dbt Analytics Engineer"],
        "github_projects": ["ETL pipeline with Airflow", "Data warehouse with dbt", "Real-time streaming with Kafka"],
        "interview_prep": ["SQL window functions", "Normalization", "Big data concepts", "Python for data manipulation"],
    },
    "Embedded & IoT": {
        "learning_path": ["Embedded Systems (Coursera — University of Colorado)", "NPTEL IoT course"],
        "practice": ["Arduino/Raspberry Pi projects", "RTOS programming", "PCB design in KiCad"],
        "companies": ["Bosch", "Honeywell", "L&T Technology", "Tata Elxsi", "Texas Instruments India"],
        "certifications": ["ARM Accredited Engineer", "Cisco IoT Fundamentals"],
        "github_projects": ["Smart home automation system", "GPS tracker with SIM800", "RTOS task scheduler"],
        "interview_prep": ["C/C++ low-level concepts", "Interrupt handling", "Communication protocols (I2C, SPI, UART)", "Memory management"],
    },
    "Mechanical/Civil CAD": {
        "learning_path": ["AutoCAD official tutorials", "SolidWorks associate prep", "ANSYS Learning Hub"],
        "practice": ["Design 5 complex assemblies", "FEA analysis project"],
        "companies": ["L&T Construction", "BHEL", "Tata Motors", "Mahindra", "Bajaj Auto"],
        "certifications": ["CSWA (SolidWorks Associate)", "AutoCAD Certified Professional"],
        "github_projects": ["CAD model library", "FEA simulation report", "Design optimization case study"],
        "interview_prep": ["Engineering drawing", "GD&T", "Material science", "Manufacturing processes"],
    },
    "Competitive Programming & Product": {
        "learning_path": ["NeetCode 150", "Striver SDE Sheet", "Grokking System Design"],
        "practice": ["LeetCode daily", "Codeforces Div 2", "Mock system design interviews"],
        "companies": ["Google", "Microsoft", "Amazon", "Flipkart", "Atlassian", "Directi"],
        "certifications": ["Google Hash Code", "ICPC participation"],
        "github_projects": ["Custom data structure implementations", "Algorithm visualizer", "Mini OS scheduler"],
        "interview_prep": ["Trees, graphs, DP", "System design (HLD/LLD)", "Behavioral rounds", "CS fundamentals"],
    },
}


def _build_user_prompt(student_dict: dict, shap_factors: list[dict], prediction_prob: float) -> str:
    top_factors = shap_factors[:5]
    factor_lines = []
    for factor in top_factors:
        feature = factor.get("feature", "Unknown feature")
        impact = factor.get("impact", 0)
        factor_lines.append(f"- {feature}: {impact}")

    if not factor_lines:
        factor_lines.append("- No SHAP factors available")

    probability_pct = round(prediction_prob * 100, 1)

    return (
        "Student profile:\n"
        f"- Branch: {student_dict.get('Branch', 'Unknown')}\n"
        f"- Gender: {student_dict.get('Gender', 'Unknown')}\n"
        f"- BTech CGPA: {student_dict.get('BTech_CGPA', 'Unknown')}\n"
        f"- Backlogs: {student_dict.get('Backlogs', 'Unknown')}\n"
        f"- Projects: {student_dict.get('No_of_Projects', 'Unknown')}\n"
        f"- Internships: {student_dict.get('Internships', 'Unknown')}\n"
        f"- Technical Skills Count: {student_dict.get('Technical_Skills_Count', 'Unknown')}\n"
        f"- Soft Skills Rating: {student_dict.get('Soft_Skills_Rating', 'Unknown')}\n"
        f"- Aptitude Score: {student_dict.get('Aptitude_Score', 'Unknown')}\n"
        f"- Placement probability: {probability_pct}%\n\n"
        "Top 5 SHAP factors influencing the result:\n"
        f"{chr(10).join(factor_lines)}\n\n"
        "Return a JSON object ONLY (no markdown, no explanation outside JSON) with this exact structure:\n"
        "{\n"
        '  "summary": "2-3 sentence honest assessment",\n'
        '  "probability_context": "what this score means for them specifically",\n'
        '  "phases": [\n'
        "    {\n"
        '      "title": "Phase 1: Month 1-2 — Foundation",\n'
        '      "focus": "one line focus area",\n'
        '      "actions": ["action 1", "action 2", "action 3", "action 4"]\n'
        "    },\n"
        "    {\n"
        '      "title": "Phase 2: Month 3-4 — Build",\n'
        '      "focus": "...",\n'
        '      "actions": [...]\n'
        "    },\n"
        "    {\n"
        '      "title": "Phase 3: Month 5-6 — Target",\n'
        '      "focus": "...",\n'
        '      "actions": [...]\n'
        "    }\n"
        "  ],\n"
        '  "quick_wins": ["thing to do this week 1", "thing to do this week 2", "thing to do this week 3"],\n'
        '  "companies_to_target": ["company tier or name relevant to their profile"],\n'
        '  "skills_to_learn": ["specific skill 1", "specific skill 2"]\n'
        "}"
    )


def _fallback_roadmap() -> dict:
    return {
        "detected_field": "Unknown",
        "summary": "Roadmap generation failed. Please try again.",
        "probability_context": "",
        "phases": [],
        "quick_wins": [],
        "companies_to_target": [],
        "skills_to_learn": [],
        "certifications": [],
        "project_ideas": [],
        "interview_prep": [],
    }


def detect_field(skills: list) -> str:
    """Detect the student's target placement field from their skills."""
    if not skills:
        return "Software Engineering (General)"

    skills_lower = [s.lower() for s in skills]

    field_keywords = {
        "AI/ML & Data Science": [
            "machine learning", "deep learning", "nlp", "natural language processing",
            "computer vision", "tensorflow", "pytorch", "scikit-learn", "pandas",
            "numpy", "data analysis", "ai/ml", "artificial intelligence", "opencv",
            "data science", "kaggle",
        ],
        "Web Development (Full Stack)": [
            "react", "node.js", "angular", "vue.js", "html/css", "javascript",
            "typescript", "django", "flask", "fastapi", "spring boot", "rest apis",
            "graphql", "full stack", "frontend", "backend",
        ],
        "Mobile Development": [
            "android", "ios", "flutter", "react native", "kotlin", "swift",
            "mobile development", "app development",
        ],
        "Cloud & DevOps": [
            "aws", "azure", "google cloud", "docker", "kubernetes", "ci/cd",
            "linux", "devops", "cloud", "terraform", "jenkins",
        ],
        "Cybersecurity": [
            "cybersecurity", "networking", "ethical hacking", "penetration testing",
            "security", "firewall", "cryptography",
        ],
        "Embedded & IoT": [
            "embedded systems", "iot", "arduino", "raspberry pi", "matlab",
            "vlsi", "pcb design", "microcontrollers", "firmware",
        ],
        "Data Engineering & Databases": [
            "sql", "mysql", "postgresql", "mongodb", "firebase", "power bi",
            "tableau", "data engineering", "spark", "hadoop", "etl",
        ],
        "Competitive Programming & Product": [
            "dsa", "competitive programming", "problem solving", "system design",
            "leetcode", "codeforces",
        ],
        "Mechanical/Civil CAD": [
            "autocad", "solidworks", "ansys", "catia", "3d modelling",
            "cad", "cam", "finite element",
        ],
    }

    scores = {}
    for field, keywords in field_keywords.items():
        score = sum(1 for kw in keywords if any(kw in s for s in skills_lower))
        if score > 0:
            scores[field] = score

    if not scores:
        return "Software Engineering (General)"

    return max(scores, key=scores.get)


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
        return None


def generate_roadmap(
    student_dict: dict,
    shap_factors: list[dict],
    prediction_prob: float,
    skills=None,
) -> dict:
    skills = skills or []
    field = detect_field(skills)
    resources = FIELD_RESOURCES.get(field, FIELD_RESOURCES["Software Engineering (General)"])

    skills_str = ", ".join(skills) if skills else "not specified"
    shap_top5 = shap_factors[:5] if shap_factors else []
    shap_str = "\n".join(
        [f"  - {f['feature']}: {'+' if f['impact'] > 0 else ''}{f['impact']:.4f}" for f in shap_top5]
    )
    if not shap_str:
        shap_str = "  - No SHAP factors available"

    prob_pct = round(prediction_prob * 100, 1)

    system_prompt = """You are a senior placement advisor for Indian engineering students.
You give hyper-specific, actionable roadmaps based on the student's exact skills and profile.
You know the Indian placement landscape deeply — service companies, product companies, startups.
Always return ONLY valid JSON. No markdown, no explanation outside the JSON object."""

    user_prompt = f"""
Student profile:
- Branch: {student_dict.get('Branch', 'CSE')}
- B.Tech CGPA: {student_dict.get('BTech_CGPA', 'N/A')}
- 10th: {student_dict.get('10th_Percentage', 'N/A')}%  |  12th: {student_dict.get('12th_Percentage', 'N/A')}%
- Projects: {student_dict.get('No_of_Projects', 0)}
- Internships: {student_dict.get('Internships', 0)}
- Backlogs: {student_dict.get('Backlogs', 0)}
- Soft Skills Rating: {student_dict.get('Soft_Skills_Rating', 'N/A')}/10
- Aptitude Score: {student_dict.get('Aptitude_Score', 'N/A')}/10
- Skills entered: {skills_str}
- Detected target field: {field}
- Placement probability: {prob_pct}%

Key factors affecting this prediction (SHAP):
{shap_str}

Field-specific resources available for this student:
- Recommended learning path: {', '.join(resources['learning_path'])}
- Practice resources: {', '.join(resources['practice'])}
- Target companies: {', '.join(resources['companies'])}
- Certifications: {', '.join(resources['certifications'])}
- Project ideas: {', '.join(resources['github_projects'])}
- Interview prep areas: {', '.join(resources['interview_prep'])}

Generate a personalized placement roadmap for this student targeting {field}.
The roadmap must be specific to their skills, address their weak SHAP factors,
and reference the actual resources listed above.

Return ONLY this JSON structure (no markdown, no text outside JSON):
{{
  "detected_field": "{field}",
  "summary": "3-4 sentence honest assessment specific to their profile and {field} field",
  "probability_context": "What {prob_pct}% means for {field} roles specifically, and what would push it higher",
  "phases": [
    {{
      "title": "Phase 1: Month 1-2 — Foundation",
      "focus": "One specific focus area for {field}",
      "actions": [
        "Specific action 1 with resource name",
        "Specific action 2 with resource name",
        "Specific action 3 with resource name",
        "Specific action 4 with resource name"
      ]
    }},
    {{
      "title": "Phase 2: Month 3-4 — Build",
      "focus": "One specific build focus for {field}",
      "actions": ["action 1", "action 2", "action 3", "action 4"]
    }},
    {{
      "title": "Phase 3: Month 5-6 — Target & Apply",
      "focus": "One specific targeting focus for {field}",
      "actions": ["action 1", "action 2", "action 3", "action 4"]
    }}
  ],
  "quick_wins": [
    "Quick win 1 — doable this week for {field}",
    "Quick win 2",
    "Quick win 3"
  ],
  "companies_to_target": {json.dumps(resources['companies'][:5])},
  "skills_to_learn": ["specific skill gap 1", "specific skill gap 2", "specific skill gap 3"],
  "certifications": {json.dumps(resources['certifications'][:2])},
  "project_ideas": {json.dumps(resources['github_projects'][:2])},
  "interview_prep": {json.dumps(resources['interview_prep'])}
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        try:
            response_text = chat_complete(messages, temperature=0.7, max_completion_tokens=1500)
            parsed = extract_json(response_text)
            if parsed is not None:
                return parsed
            logger.warning("Roadmap JSON parsing failed on attempt %s", attempt + 1)
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"{user_prompt}\n\nReturn ONLY valid JSON, nothing else.",
                    },
                ]
            else:
                return _fallback_roadmap()
        except Exception as e:
            logger.exception("Roadmap generation failed on attempt %s", attempt + 1)
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"{user_prompt}\n\nReturn ONLY valid JSON, nothing else.",
                    },
                ]
            else:
                st.error(f"Debug error: {e}")
                return _fallback_roadmap()

    return _fallback_roadmap()
