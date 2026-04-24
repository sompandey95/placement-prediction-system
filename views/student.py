import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import dotenv_values

try:
    import shap
except ImportError:
    shap = None

from src.evaluate import load_comparison_results
from src.predict import load_artifacts, predict_student
from src.roadmap_gen import generate_roadmap
from src.roadmap_pdf import generate_roadmap_pdf


FEATURE_LABELS = {
    "BTech_CGPA": "B.Tech CGPA",
    "10th_Percentage": "10th Percentage",
    "12th_Percentage": "12th Percentage",
    "No_of_Projects": "No. of Projects",
    "Technical_Skills_Count": "Technical Skills",
    "Soft_Skills_Rating": "Soft Skills Rating",
    "Aptitude_Score": "Aptitude Score",
    "Backlogs": "Backlogs",
    "Internships": "Internships",
    "Gender_Male": "Gender (Male)",
    "Branch_CSE": "Branch: CSE",
    "Branch_IT": "Branch: IT",
    "Branch_ECE": "Branch: ECE",
    "Branch_EEE": "Branch: EEE",
    "Branch_ME": "Branch: ME",
    "Branch_CIVIL": "Branch: Civil",
}


SKILL_OPTIONS = [
    "Python", "Java", "C", "C++", "JavaScript", "TypeScript", "Go", "Rust", "Kotlin", "Swift",
    "HTML/CSS", "React", "Node.js", "Vue.js", "Angular", "Django", "Flask", "FastAPI", "Spring Boot",
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "OpenCV",
    "Data Analysis", "Power BI", "Tableau", "SQL", "MySQL", "PostgreSQL", "MongoDB", "Firebase",
    "Android Development", "iOS Development", "Flutter", "React Native",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "CI/CD", "Linux",
    "Cybersecurity", "Networking", "Embedded Systems", "IoT", "Arduino", "MATLAB",
    "AutoCAD", "SolidWorks", "VLSI", "PCB Design",
    "Git", "GitHub", "REST APIs", "GraphQL", "Microservices", "System Design",
    "DSA", "Competitive Programming", "Problem Solving",
]


model, feature_cols = load_artifacts()
comparison_df = load_comparison_results()


def _escape_html(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_shap_inline_table(shap_df):
    if shap_df is None or shap_df.empty:
        st.warning("SHAP explanation unavailable.")
        return

    shap_top = shap_df.head(8).copy()
    max_abs = shap_top["SHAP Value"].abs().max()
    if max_abs == 0:
        max_abs = 1

    rows = []
    for _, row in shap_top.iterrows():
        raw_feature_name = row["Feature"]
        display_name = FEATURE_LABELS.get(
            raw_feature_name, raw_feature_name.replace("_", " ").title()
        )
        value = float(row["SHAP Value"])
        width = abs(value) / max_abs * 100
        impact = "Positive" if value > 0 else "Negative"
        fill_class = "shap-bar-fill-pos" if value > 0 else "shap-bar-fill-neg"
        badge_bg = "#14532d" if value > 0 else "#7f1d1d"
        badge_color = "#bbf7d0" if value > 0 else "#fecaca"
        rows.append(
            f"""
            <div class="shap-inline-row">
              <div class="shap-feat-name">{_escape_html(display_name)}</div>
              <div class="shap-bar-bg"><div class="{fill_class}" style="width:{width:.1f}%"></div></div>
              <div class="shap-val-text">{value:.4f}</div>
              <div style="font-size:11px;padding:2px 8px;border-radius:999px;background:{badge_bg};color:{badge_color};">
                {impact}
              </div>
            </div>
            """
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


def render_floating_chat(student_dict: dict, prob: float, skills: list, pred: int):
    """
    Renders a fully self-contained floating chat widget using st.components.v1.html.
    Calls Azure OpenAI directly from JavaScript. No Streamlit rerun needed.
    """
    import json

    api_key = st.secrets.get("AZURE_OPENAI_MINI_API_KEY") or os.getenv("AZURE_OPENAI_MINI_API_KEY", "")
    endpoint = (st.secrets.get("AZURE_OPENAI_MINI_ENDPOINT") or os.getenv("AZURE_OPENAI_MINI_ENDPOINT", "")).rstrip("/")
    api_version = st.secrets.get("AZURE_OPENAI_MINI_API_VERSION") or os.getenv("AZURE_OPENAI_MINI_API_VERSION", "2025-04-01-preview")
    deployment = st.secrets.get("AZURE_OPENAI_MINI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-4o-mini")

    result_text = "Likely to be Placed" if pred == 1 else "At Risk - Needs Improvement"
    skills_str = ", ".join(skills) if skills else "Not specified"
    prob_pct = round(prob * 100, 1)

    system_prompt = f"""You are a highly knowledgeable placement advisor for Indian engineering students. You have the student's complete profile below and must always refer to their specific details.

STUDENT PROFILE:
- Branch: {student_dict.get('Branch', 'CSE')}
- B.Tech CGPA: {student_dict.get('BTech_CGPA', 'N/A')}
- 10th / 12th: {student_dict.get('10th_Percentage', 'N/A')}% / {student_dict.get('12th_Percentage', 'N/A')}%
- Projects: {student_dict.get('No_of_Projects', 0)}
- Internships: {student_dict.get('Internships', 0)}
- Backlogs: {student_dict.get('Backlogs', 0)}
- Technical Skills: {skills_str}
- Soft Skills Rating: {student_dict.get('Soft_Skills_Rating', 'N/A')}/10
- Aptitude Score: {student_dict.get('Aptitude_Score', 'N/A')}/10
- Placement Prediction: {result_text}
- Placement Probability: {prob_pct}%

RESPONSE FORMATTING RULES - follow these strictly:
1. Structure every response with clear sections when answering multi-part questions
2. Use numbered lists for step-by-step plans or ordered actions
3. Use bullet points (-) for feature lists or unordered items
4. Bold important terms, company names, and key actions using **text**
5. Keep section headers short, on their own line, followed by a colon
6. Never write walls of text - break into digestible chunks
7. End every response with one specific next action the student should take TODAY

RESPONSE LENGTH RULES:
- Simple questions (yes/no, single fact): 2-3 sentences max
- Advice questions: 1 short intro + structured list + 1 closing action
- Plan requests (30-day, 90-day): Full structured plan with phases
- Never exceed 5 major points unless a detailed plan is explicitly requested

CONTENT RULES:
- Always reference the student's ACTUAL numbers (their CGPA, their projects count, etc.)
- Mention specific Indian companies relevant to their branch and skills
- Reference specific resources by name (NeetCode, InterviewBit, Andrew Ng, etc.)
- For probability improvement questions: give specific, quantified advice
- Never give generic advice that ignores their profile"""

    system_prompt_js = json.dumps(system_prompt)
    azure_url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}

  #chat-bubble {{
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 54px;
    height: 54px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F8BF9, #7B61FF);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10000;
    box-shadow: 0 4px 16px rgba(79,139,249,0.5);
    border: none;
    transition: transform 0.2s ease;
    font-size: 24px;
  }}
  #chat-bubble:hover {{ transform: scale(1.08); }}

  #chat-panel {{
    position: fixed;
    bottom: 94px;
    right: 28px;
    width: 360px;
    height: 480px;
    min-width: 280px;
    min-height: 320px;
    max-width: 600px;
    max-height: 80vh;
    background: #1c1c2e;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    z-index: 9999;
    display: none;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    resize: both;
  }}
  #chat-panel.open {{ display: flex; }}

  #chat-header {{
    background: linear-gradient(135deg, #4F8BF9, #7B61FF);
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    border-radius: 18px 18px 0 0;
  }}
  #chat-header-left {{ display: flex; align-items: center; gap: 8px; }}
  #chat-header-avatar {{
    width: 32px; height: 32px; border-radius: 50%;
    background: rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
  }}
  #chat-header-text {{ color: white; }}
  #chat-header-text .title {{ font-size: 13px; font-weight: 600; }}
  #chat-header-text .sub {{ font-size: 10px; opacity: 0.8; }}
  #chat-header-right {{ display: flex; gap: 6px; }}
  .header-btn {{
    background: rgba(255,255,255,0.15);
    border: none; cursor: pointer; color: white;
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; transition: background 0.2s;
  }}
  .header-btn:hover {{ background: rgba(255,255,255,0.28); }}

  #chat-context-bar {{
    background: rgba(79,139,249,0.12);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 6px 14px;
    font-size: 10px;
    color: #7EB8F7;
    flex-shrink: 0;
  }}

  #chat-messages {{
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.1) transparent;
  }}
  #chat-messages::-webkit-scrollbar {{ width: 4px; }}
  #chat-messages::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 2px; }}

  .msg-row {{ display: flex; align-items: flex-end; gap: 6px; }}
  .msg-row.user {{ flex-direction: row-reverse; }}

  .msg-avatar {{
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0;
  }}
  .msg-avatar.ai {{ background: linear-gradient(135deg,#4F8BF9,#7B61FF); color:white; }}
  .msg-avatar.user {{ background: rgba(255,255,255,0.1); color:white; }}

  .msg-bubble {{
    max-width: 88%;
    padding: 9px 12px;
    border-radius: 14px;
    font-size: 13px;
    line-height: 1.5;
    color: #e8e8f0;
    word-break: break-word;
  }}
  .msg-bubble strong {{ color: #e8f4ff; }}
  .msg-bubble em {{ color: #b8d4f0; font-style: italic; }}
  .msg-bubble.ai br {{ display: block; margin: 1px 0; }}
  .msg-bubble.ai div {{ margin: 1px 0; }}
  .msg-bubble.ai {{
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom-left-radius: 4px;
  }}
  .msg-bubble.user {{
    background: linear-gradient(135deg,#4F8BF9,#7B61FF);
    color: white;
    border-bottom-right-radius: 4px;
  }}

  .typing-indicator {{
    display: flex; gap: 4px; align-items: center; padding: 4px 2px;
  }}
  .typing-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: #7EB8F7;
    animation: typing-bounce 1.2s infinite ease-in-out;
  }}
  .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
  .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
  @keyframes typing-bounce {{
    0%, 80%, 100% {{ transform: translateY(0); opacity: 0.4; }}
    40% {{ transform: translateY(-6px); opacity: 1; }}
  }}

  #chat-input-area {{
    padding: 10px 12px;
    border-top: 1px solid rgba(255,255,255,0.08);
    display: flex;
    gap: 8px;
    align-items: flex-end;
    flex-shrink: 0;
    background: rgba(255,255,255,0.02);
  }}
  #chat-input {{
    flex: 1;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 8px 14px;
    color: white;
    font-size: 12.5px;
    resize: none;
    max-height: 80px;
    min-height: 36px;
    outline: none;
    font-family: inherit;
    line-height: 1.4;
  }}
  #chat-input::placeholder {{ color: rgba(255,255,255,0.35); }}
  #chat-input:focus {{ border-color: #4F8BF9; }}

  #send-btn {{
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg,#4F8BF9,#7B61FF);
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: transform 0.15s, opacity 0.15s;
  }}
  #send-btn:hover {{ transform: scale(1.08); }}
  #send-btn:disabled {{ opacity: 0.4; cursor: not-allowed; transform: none; }}
  #send-btn svg {{ width: 16px; height: 16px; fill: white; }}

  #clear-btn {{
    background: none; border: none; cursor: pointer;
    color: rgba(255,255,255,0.3); font-size: 10px;
    padding: 2px 4px; flex-shrink: 0;
    transition: color 0.2s;
  }}
  #clear-btn:hover {{ color: rgba(255,255,255,0.6); }}
</style>
</head>
<body>

<button id="chat-bubble" onclick="togglePanel()" title="Open Placement Advisor">🤖</button>

<div id="chat-panel">
  <div id="chat-header">
    <div id="chat-header-left">
      <div id="chat-header-avatar">🤖</div>
      <div id="chat-header-text">
        <div class="title">Placement Advisor</div>
        <div class="sub">AI-powered · knows your profile</div>
      </div>
    </div>
    <div id="chat-header-right">
      <button class="header-btn" onclick="clearChat()" title="Clear chat">🗑</button>
      <button class="header-btn" onclick="togglePanel()" title="Close">✕</button>
    </div>
  </div>

  <div id="chat-context-bar">
    📊 {prob_pct}% placement probability · {student_dict.get('Branch','CSE')} · CGPA {student_dict.get('BTech_CGPA','N/A')} · {result_text}
  </div>

  <div id="chat-messages">
    <div class="msg-row">
      <div class="msg-avatar ai">🤖</div>
      <div class="msg-bubble ai">Hi! I'm your placement advisor. I have your full profile — your {student_dict.get('Branch','CSE')} background, CGPA of {student_dict.get('BTech_CGPA','N/A')}, and {prob_pct}% placement probability. Ask me anything about placements, interview prep, or improving your chances! 💪</div>
    </div>
  </div>

  <div id="chat-input-area">
    <textarea id="chat-input" placeholder="Ask anything about placements..." rows="1"
      onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button id="send-btn" onclick="sendMessage()" title="Send">
      <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/></svg>
    </button>
    <button id="clear-btn" onclick="clearChat()" title="Clear">🗑</button>
  </div>
</div>

<script>
  const AZURE_URL = {json.dumps(azure_url)};
  const API_KEY = {json.dumps(api_key)};
  const SYSTEM_PROMPT = {system_prompt_js};

  let conversationHistory = [
    {{ role: "system", content: SYSTEM_PROMPT }}
  ];

  let isOpen = false;
  let isLoading = false;

  function togglePanel() {{
    isOpen = !isOpen;
    const panel = document.getElementById('chat-panel');
    if (isOpen) {{
      panel.classList.add('open');
      document.getElementById('chat-input').focus();
    }} else {{
      panel.classList.remove('open');
    }}
  }}

  function autoResize(el) {{
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 80) + 'px';
  }}

  function handleKey(e) {{
    if (e.key === 'Enter' && !e.shiftKey) {{
      e.preventDefault();
      sendMessage();
    }}
  }}

  function scrollToBottom() {{
    const msgs = document.getElementById('chat-messages');
    msgs.scrollTop = msgs.scrollHeight;
  }}

  function formatMessage(text) {{
    text = text
      .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
      .replace(/__(.*?)__/g, '<strong>$1</strong>')
      .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:1px 5px;border-radius:3px;font-size:11px;">$1</code>')
      .replace(/^(\\d+)\\.\\s(.+)$/gm, '<div style="padding:2px 0 2px 8px;"><span style="color:#7EB8F7;font-weight:600;min-width:18px;display:inline-block;">$1.</span> $2</div>')
      .replace(/^[-*]\\s(.+)$/gm, '<div style="padding:2px 0 2px 8px;">• $1</div>')
      .replace(/^([A-Z][^:\\n]{{2,35}}):\\s*$/gm, '<div style="color:#7EB8F7;font-weight:600;margin-top:8px;margin-bottom:2px;">$1</div>')
      .replace(/\\n\\n/g, '<br><br>')
      .replace(/\\n/g, '<br>');
    return text;
  }}

  function appendMessage(role, content) {{
    const msgs = document.getElementById('chat-messages');
    const row = document.createElement('div');
    row.className = `msg-row ${{role}}`;
    const avatar = document.createElement('div');
    avatar.className = `msg-avatar ${{role}}`;
    avatar.textContent = role === 'ai' ? '🤖' : '🧑';
    const bubble = document.createElement('div');
    bubble.className = `msg-bubble ${{role}}`;
    bubble.innerHTML = formatMessage(content);
    row.appendChild(avatar);
    row.appendChild(bubble);
    msgs.appendChild(row);
    scrollToBottom();
    return bubble;
  }}

  function showTyping() {{
    const msgs = document.getElementById('chat-messages');
    const row = document.createElement('div');
    row.className = 'msg-row';
    row.id = 'typing-row';
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar ai';
    avatar.textContent = '🤖';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble ai';
    bubble.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
    row.appendChild(avatar);
    row.appendChild(bubble);
    msgs.appendChild(row);
    scrollToBottom();
  }}

  function removeTyping() {{
    const el = document.getElementById('typing-row');
    if (el) el.remove();
  }}

  function clearChat() {{
    conversationHistory = [{{ role: "system", content: SYSTEM_PROMPT }}];
    const msgs = document.getElementById('chat-messages');
    msgs.innerHTML = '';
    appendMessage('ai', "Chat cleared! Ask me anything about your placement preparation.");
  }}

  async function sendMessage() {{
    if (isLoading) return;
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    input.style.height = 'auto';
    appendMessage('user', text);

    conversationHistory.push({{ role: "user", content: text }});
    if (conversationHistory.length > 22) {{
      conversationHistory = [conversationHistory[0], ...conversationHistory.slice(-20)];
    }}

    isLoading = true;
    document.getElementById('send-btn').disabled = true;
    showTyping();

    try {{
      const response = await fetch(AZURE_URL, {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json',
          'api-key': API_KEY
        }},
        body: JSON.stringify({{
          messages: conversationHistory,
          max_completion_tokens: 600,
          temperature: 0.7
        }})
      }});

      if (!response.ok) {{
        const err = await response.json();
        throw new Error(err.error?.message || 'API error');
      }}

      const data = await response.json();
      const reply = data.choices[0].message.content;
      conversationHistory.push({{ role: "assistant", content: reply }});

      removeTyping();
      appendMessage('ai', reply);

    }} catch(err) {{
      removeTyping();
      appendMessage('ai', 'Sorry, something went wrong: ' + err.message + '. Please try again.');
      conversationHistory.pop();
    }}

    isLoading = false;
    document.getElementById('send-btn').disabled = false;
    document.getElementById('chat-input').focus();
  }}
</script>
</body>
</html>
"""
     # height=660 so iframe renders fully; CSS in show() makes it position:fixed overlay
    components.html(html_code, height=660, scrolling=False)

def show():
    st.markdown(
        """
        <style>
        /* Remove default Streamlit top padding */
        .block-container { padding-top: 1.5rem !important; }

        /* Fix title overflow */
        h1, h2, h3 {
          white-space: normal !important;
          overflow: visible !important;
          text-overflow: unset !important;
          word-wrap: break-word !important;
        }

        [data-testid="stAppViewBlockContainer"] {
          overflow: visible !important;
        }

        .stApp header, [data-testid="stHeader"] {
          overflow: visible !important;
        }

        /* Metric cards row */
        .metric-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
        .metric-card { background: rgba(255,255,255,0.04); border: 0.5px solid rgba(255,255,255,0.1);
          border-radius: 10px; padding: 14px 16px; }
        .metric-card .mc-label { font-size: 12px; color: #9ca3af; margin-bottom: 4px; }
        .metric-card .mc-val { font-size: 22px; font-weight: 500; }

        /* Card wrapper for any section */
        .section-card { background: rgba(255,255,255,0.03); border: 0.5px solid rgba(255,255,255,0.08);
          border-radius: 12px; padding: 20px; margin-bottom: 16px; }

        /* Hide Streamlit's default chat input sticky bar styling */
        .stChatInput { border-radius: 20px !important; }

        /* Float the chat over the viewport instead of inside page flow. */
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

        /* Collapse layout space — targets direct parent reliably across Streamlit versions */
        div:has(> [data-testid="stCustomComponentV1"]),
        div:has(> div > [data-testid="stCustomComponentV1"]) {
          height: 0px !important;
          min-height: 0px !important;
          max-height: 0px !important;
          overflow: visible !important;
          padding: 0 !important;
          margin: 0 !important;
        }

        /* SHAP inline bar chart rows */
        .shap-inline-row { display: flex; align-items: center; gap: 10px;
          padding: 5px 0; border-bottom: 0.5px solid rgba(255,255,255,0.06); }
        .shap-inline-row:last-child { border-bottom: none; }
        .shap-feat-name { font-size: 12px; flex: 1.2; }
        .shap-bar-bg { flex: 2; height: 7px; background: rgba(255,255,255,0.06);
          border-radius: 4px; overflow: hidden; }
        .shap-bar-fill-pos { height: 100%; background: #22c55e; border-radius: 4px; }
        .shap-bar-fill-neg { height: 100%; background: #ef4444; border-radius: 4px; }
        .shap-val-text { font-size: 11px; color: #9ca3af; min-width: 44px; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-bottom:18px;">
          <h2 style="margin:0 0 4px 0;font-size:clamp(18px, 3vw, 32px);
            font-weight:650; white-space:normal; word-wrap:break-word; line-height:1.3;">
            🎓 Student Placement Prediction
          </h2>
          <div style="color:#9ca3af;font-size:14px;">Fill in your details and predict your placement likelihood</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ About the Prediction Model"):
        st.write("The system trained 6 ML models and selected the best by F1 score.")
        best_model_name = comparison_df.loc[comparison_df["f1"].idxmax(), "model_name"]
        model_info_df = comparison_df[["model_name", "accuracy", "f1", "roc_auc", "cv_mean"]]
        styled_df = model_info_df.style.highlight_max(subset=["f1"], color="#d5f5e3").format(
            {
                "accuracy": "{:.4f}",
                "f1": "{:.4f}",
                "roc_auc": "{:.4f}",
                "cv_mean": "{:.4f}",
            }
        )
        st.dataframe(styled_df, use_container_width=True)
        st.success(f"✅ Active model: {best_model_name}")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenth = st.number_input("10th Percentage", 0.0, 100.0, 75.0, step=0.1)
        twelfth = st.number_input("12th Percentage", 0.0, 100.0, 75.0, step=0.1)
        cgpa = st.number_input("BTech CGPA", 0.0, 10.0, 7.5, step=0.1)
        projects = st.number_input("Number of Projects", 0, 10, 2)

    with col2:
        internships = st.number_input("Internships Done", 0, 5, 0)
        st.markdown("**Technical Skills**")

        selected_skills = st.multiselect(
            "Select your skills (type to search)",
            options=SKILL_OPTIONS,
            default=st.session_state.get("selected_skills", []),
            key="skills_multiselect",
            help="Select all skills you are comfortable with",
        )

        custom_skill = st.text_input(
            "Add a custom skill not in the list",
            value="",
            key="custom_skill_input",
            placeholder="e.g. Solidity, ROS, Blender...",
        )

        if custom_skill.strip():
            all_skills = selected_skills + [s.strip() for s in custom_skill.split(",") if s.strip()]
        else:
            all_skills = selected_skills

        if all_skills:
            st.caption(
                f"✅ {len(all_skills)} skill(s) selected: {', '.join(all_skills[:5])}"
                f"{'...' if len(all_skills) > 5 else ''}"
            )
        else:
            st.caption("⚠️ Select at least 1 skill for a better roadmap")

        technical_skills_count = len(all_skills) if all_skills else 1
        st.session_state["selected_skills"] = all_skills
        soft_skills = st.slider("Soft Skills Rating", 1, 10, 7)
        aptitude = st.slider("Aptitude Score", 1, 10, 6)

    with col3:
        backlogs = st.number_input("Number of Backlogs", 0, 10, 0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EEE", "IT"])
        st.empty()
        st.markdown("---")
        predict_btn = st.button("🔍 Predict Placement", use_container_width=True)

    if predict_btn:
        if "roadmap_result" in st.session_state:
            del st.session_state["roadmap_result"]
        st.session_state["student_skills"] = all_skills

        student_dict = {
            "Gender": gender,
            "Branch": branch,
            "10th_Percentage": tenth,
            "12th_Percentage": twelfth,
            "BTech_CGPA": cgpa,
            "No_of_Projects": projects,
            "Internships": internships,
            "Technical_Skills_Count": technical_skills_count,
            "Soft_Skills_Rating": soft_skills,
            "Backlogs": backlogs,
            "Aptitude_Score": aptitude,
        }

        prob, pred = predict_student(student_dict, model, feature_cols)
        st.session_state["latest_prediction"] = {
            "student_dict": student_dict,
            "prob": prob,
            "pred": pred,
        }

    if "latest_prediction" in st.session_state:
        prediction_context = st.session_state["latest_prediction"]
        student_dict = prediction_context["student_dict"]
        prob = prediction_context["prob"]
        pred = prediction_context["pred"]

        shap_df = None
        try:
            import shap
            from src.preprocess import load_and_preprocess

            sdf = pd.DataFrame([student_dict])
            sdf = pd.get_dummies(sdf, drop_first=True)
            for col in feature_cols:
                if col not in sdf.columns:
                    sdf[col] = 0
            sdf = sdf[feature_cols]

            X_train, X_test, y_train, y_test, _ = load_and_preprocess()
            background = pd.DataFrame(X_train, columns=feature_cols)

            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(background, 50),
            )
            shap_values = explainer.shap_values(sdf, nsamples=100)

            if isinstance(shap_values, list):
                sv_array = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                sv_array = shap_values[0, :, 1]
            else:
                sv_array = shap_values[0]

            shap_df = pd.DataFrame(
                {
                    "Feature": feature_cols,
                    "SHAP Value": sv_array,
                }
            )
            shap_df["SHAP Value"] = shap_df["SHAP Value"].round(4)
            shap_df = (
                shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index)
                .head(10)
                .reset_index(drop=True)
            )
            shap_df["Impact"] = shap_df["SHAP Value"].apply(
                lambda x: "Positive 🟢" if x > 0 else "Negative 🔴"
            )
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
            shap_df = None

        # Key risk factor — cleaned display name via FEATURE_LABELS
        key_risk = "Unavailable"
        if shap_df is not None and not shap_df.empty and (shap_df["SHAP Value"] < 0).any():
            key_risk_raw = shap_df.loc[shap_df["SHAP Value"].idxmin(), "Feature"]
            key_risk = FEATURE_LABELS.get(
                key_risk_raw,
                key_risk_raw.replace("_", " ").title()
            )

        result_col, prob_col, risk_col = st.columns(3)
        with result_col:
            st.metric("Placement Result", "✅ Likely Placed" if pred == 1 else "❌ At Risk")
        with prob_col:
            st.metric("Probability", f"{prob * 100:.1f}%")
        with risk_col:
            st.metric("Key Risk Factor", key_risk)
            st.caption("via SHAP")

        col_gauge, col_radar = st.columns([1, 1.4])

        with col_gauge:
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    delta={"reference": 50},
                    title={"text": "Placement Probability %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "green" if prob >= 0.5 else "red"},
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            gauge_fig.update_layout(
                height=200,
                margin=dict(t=10, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_radar:
            radar_data = pd.DataFrame(
                {
                    "Category": [
                        "10th", "12th", "CGPA", "Projects",
                        "Tech Skills", "Soft Skills", "Aptitude",
                    ],
                    "Score": [
                        student_dict["10th_Percentage"] / 10,
                        student_dict["12th_Percentage"] / 10,
                        student_dict["BTech_CGPA"],
                        student_dict["No_of_Projects"],
                        student_dict["Technical_Skills_Count"],
                        student_dict["Soft_Skills_Rating"],
                        student_dict["Aptitude_Score"],
                    ],
                }
            )
            radar_fig = px.line_polar(
                radar_data,
                r="Score",
                theta="Category",
                line_close=True,
            )
            radar_fig.update_traces(fill="toself")
            radar_fig.update_layout(height=220, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(radar_fig, use_container_width=True)

        st.subheader("🔍 Why this prediction? (SHAP Explanation)")
        st.info(
            "SHAP values show how much each feature pushed the prediction toward or away from placement."
        )
        _render_shap_inline_table(shap_df)

        st.subheader("🗺️ Your Personalized Placement Roadmap")

        if st.button("Generate My Placement Roadmap"):
            shap_factors = []
            if shap_df is not None and not shap_df.empty:
                shap_factors = [
                    {
                        "feature": row["Feature"],
                        "impact": float(row["SHAP Value"]),
                    }
                    for _, row in shap_df.iterrows()
                ]

            with st.spinner("Generating your roadmap with AI..."):
                try:
                    st.session_state["roadmap_result"] = generate_roadmap(
                        student_dict,
                        shap_factors,
                        prob,
                        skills=st.session_state.get("student_skills", []),
                    )
                except TypeError as e:
                    if "unexpected keyword argument 'skills'" not in str(e):
                        raise
                    st.session_state["roadmap_result"] = generate_roadmap(
                        student_dict,
                        shap_factors,
                        prob,
                    )

        roadmap = st.session_state.get("roadmap_result")
        if roadmap:
            st.info(roadmap.get("summary", "No summary available."))
            st.caption(roadmap.get("probability_context", ""))

            phases = roadmap.get("phases", []) or []
            ph1, ph2, ph3 = st.columns(3)
            for col, phase in zip([ph1, ph2, ph3], phases[:3]):
                with col:
                    st.markdown(
                        f"""
                        <div style="background:rgba(255,255,255,0.04);border:0.5px solid rgba(255,255,255,0.1);
                          border-radius:10px;padding:14px;height:100%">
                          <div style="font-size:12px;font-weight:500;color:#60a5fa;margin-bottom:4px">
                            {phase.get('title', '')}</div>
                          <div style="font-size:11px;color:#9ca3af;font-style:italic;margin-bottom:10px">
                            {phase.get('focus', '')}</div>
                          {"".join(f'<div style="font-size:12px;padding:3px 0;padding-left:12px;position:relative">• {a}</div>' for a in phase.get('actions', []))}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            qw_col, co_col, sk_col = st.columns(3)
            with qw_col:
                st.markdown("**⚡ Quick Wins**")
                for item in roadmap.get("quick_wins", []):
                    st.markdown(f"- {item}")
            with co_col:
                st.markdown("**🎯 Companies to Target**")
                for item in roadmap.get("companies_to_target", []):
                    st.markdown(f"- {item}")
            with sk_col:
                st.markdown("**🛠️ Skills to Learn**")
                for item in roadmap.get("skills_to_learn", []):
                    st.markdown(f"- {item}")

            student_name = student_dict.get("Name", "Student")
            roadmap_pdf = generate_roadmap_pdf(student_name, roadmap)
            st.download_button(
                label="📄 Download Roadmap PDF",
                data=roadmap_pdf,
                file_name="placement_roadmap.pdf",
                mime="application/pdf",
            )

        st.subheader("📄 Download Your Report")
        try:
            from src.report_gen import generate_report

            pdf_bytes = generate_report(student_dict, prob, pred, shap_df)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="placement_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF generation unavailable: {e}")

        st.subheader("💡 Personalized Recommendations")
        if student_dict["Backlogs"] > 0:
            st.warning(
                "📚 Clear your backlogs. Backlogs are among the strongest negative signals in placement."
            )
        if student_dict["No_of_Projects"] < 2:
            st.warning(
                "🛠️ Work on at least 2-3 projects. Project count is a top feature in placement prediction."
            )
        if student_dict["BTech_CGPA"] < 7.0:
            st.warning(
                "📈 Aim to improve your CGPA above 7.0. Academic performance significantly affects shortlisting."
            )
        if student_dict["Technical_Skills_Count"] < 5:
            st.warning("💻 Add more technical skills. Aim for at least 5 certifiable skills.")
        if student_dict["Internships"] == 0:
            st.info("🏢 Try to complete at least one internship before placement season.")
        if student_dict["Aptitude_Score"] < 6:
            st.info("🧠 Practice aptitude tests regularly. Many companies filter on aptitude scores.")
        if pred == 1 and student_dict["Backlogs"] == 0 and student_dict["No_of_Projects"] >= 2:
            st.success("🎯 You have a strong profile! Focus on interview preparation now.")
