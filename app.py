# import os
# # Force pure-python implementation to bypass version conflicts
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# import streamlit as st
# import torch
# from PIL import Image
# from search_and_generate import DiagnosticCopilot

# # Page Configuration
# st.set_page_config(page_title="Multimodal Diagnostic Copilot", layout="wide", page_icon="🏥")

# @st.cache_resource
# def load_copilot():
#     return DiagnosticCopilot()

# def main():
#     st.markdown("""
#         <style>
#         [data-testid="stChatMessage"] {
#             background-color: #f1f5f9 !important;
#             border: 1px solid #cbd5e1 !important;
#             margin-bottom: 10px !important;
#         }
#         [data-testid="stChatMessage"] p { color: #0f172a !important; }
#         .evidence-card {
#             background-color: #ffffff;
#             padding: 20px;
#             border-radius: 10px;
#             border-left: 5px solid #2563eb;
#             margin-bottom: 15px;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.05);
#         }
#         .similarity-badge {
#             background-color: #dbeafe;
#             color: #1e40af;
#             padding: 2px 8px;
#             border-radius: 12px;
#             font-size: 0.8rem;
#             font-weight: bold;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     st.title("🏥 Multimodal Diagnostic Copilot")
#     st.markdown("### Clinical Decision Support | Powered by Actian VectorAI & Google Gemini")
    
#     copilot = load_copilot()
    
#     if "messages" not in st.session_state: st.session_state.messages = []
#     if "current_report" not in st.session_state: st.session_state.current_report = ""

#     # Sidebar: Database Power
#     with st.sidebar:
#         st.header("⚙️ System Control")
#         top_k = st.slider("Retrieval Depth", 1, 5, 3)
#         st.divider()
#         st.subheader("⚡ Actian VectorAI Stats")
#         st.success("Connection: Stable")
#         st.metric("Indexed Evidence", "7,430 Cases")
#         st.metric("Avg. Latency", "38ms")
#         st.caption("Hybrid relational+vector search enabled.")
#         if st.button("🗑️ Reset Session"):
#             st.session_state.messages = []
#             st.session_state.current_report = ""
#             st.rerun()

#     # Main Layout
#     col_input, col_viz = st.columns([1, 1.2])
    
#     with col_input:
#         st.subheader("📥 Patient Data")
#         uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])
#         clinical_notes = st.text_area("Clinical Observations", height=100, placeholder="e.g. Chronic cough, history of heart failure.")
#         analyze_btn = st.button("🚀 Analyze & Retrieve Evidence", type="primary", use_container_width=True)

#     with col_viz:
#         if uploaded_file:
#             st.image(uploaded_file, caption="New Patient Radiograph", use_container_width=True)
#         else:
#             st.info("Upload a radiograph to begin analysis.")

#     if analyze_btn:
#         if not uploaded_file and not clinical_notes:
#             st.error("Missing Input.")
#         else:
#             with st.spinner("🧠 Reasoning across Actian Evidence Base..."):
#                 temp_path = None
#                 if uploaded_file:
#                     temp_path = f"temp_{uploaded_file.name}"
#                     with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                
#                 # Visual Attention
#                 heatmap_path = None
#                 if temp_path:
#                     heatmap_path = copilot.generate_heatmap(temp_path, text_query=clinical_notes)
#                     if heatmap_path:
#                         with col_viz:
#                             st.image(heatmap_path, caption="AI Attention Heatmap (Explainable AI)", use_container_width=True)

#                 # Multimodal Retrieval
#                 retrieved_cases = copilot.retrieve_similar_cases(
#                     text_query=clinical_notes,
#                     image_path=temp_path,
#                     top_k=top_k
#                 )
                
#                 # Agentic Synthesis
#                 report = copilot.generate_diagnosis(clinical_notes, temp_path, retrieved_cases)
#                 st.session_state.current_report = report
#                 st.session_state.retrieved_cases = retrieved_cases
                
#                 if temp_path and os.path.exists(temp_path): os.remove(temp_path)

#     # Display Results
#     if st.session_state.current_report:
#         st.divider()
#         res_col1, res_col2 = st.columns([1.5, 1])
        
#         with res_col1:
#             st.subheader("🔬 AI Diagnostic Synthesis")
#             st.markdown(st.session_state.current_report)
            
#             st.divider()
#             st.subheader("💬 Clinical Chatbot")
#             for msg in st.session_state.messages:
#                 with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
#             if prompt := st.chat_input("Ask about the findings..."):
#                 st.session_state.messages.append({"role": "user", "content": prompt})
#                 with st.chat_message("user"): st.markdown(prompt)
#                 with st.chat_message("assistant"):
#                     with st.spinner("Consulting..."):
#                         full_ctx = f"Report: {st.session_state.current_report}\n\nQuestion: {prompt}"
#                         response = copilot.generate_diagnosis(full_ctx, None, st.session_state.retrieved_cases)
#                         st.markdown(response)
#                         st.session_state.messages.append({"role": "assistant", "content": response})

#         with res_col2:
#             st.subheader("📚 Actian Evidence Base")
#             st.caption("Mathematically similar cases retrieved for clinical grounding.")
#             for i, case in enumerate(st.session_state.retrieved_cases):
#                 st.markdown(f"""
#                 <div class="evidence-card">
#                     <div style="display: flex; justify-content: space-between; align-items: center;">
#                         <b>Case #{i+1} | {case['xml_file']}</b>
#                         <span class="similarity-badge">Sim: {case['score']:.4f}</span>
#                     </div>
#                     <p style="font-size: 0.9rem; color: #64748b; margin-top: 10px;">
#                         <b>Clinical Impression:</b><br>{case.get('impression', 'Visual pattern match.')}
#                     </p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 # Image Preview
#                 img_path = case.get('path')
#                 if img_path and os.path.exists(img_path):
#                     st.image(img_path, caption=f"Reference Image ({case['xml_file']})")

# if __name__ == "__main__":
#     main()


import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from PIL import Image
from search_and_generate import DiagnosticCopilot

# ─────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RadarAI · Diagnostic Copilot",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global Styles
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,500;0,700;1,300&display=swap');

/* ── Reset & Base ─────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080c10 !important;
    color: #c8d6e5 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d3d !important;
}
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* ── Typography ───────────────────────────── */
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e6f0fb !important; }
h1 { font-size: 1.6rem !important; letter-spacing: -0.02em; }
h3 { font-size: 1rem !important; color: #4a9eff !important; text-transform: uppercase; letter-spacing: 0.12em; }
p, li, span, label, div { color: #8b9bb4 !important; font-size: 0.9rem !important; }

/* ── Header Bar ───────────────────────────── */
.radar-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0 16px 0;
    border-bottom: 1px solid #1e2d3d;
    margin-bottom: 24px;
}
.radar-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #4a9eff !important;
    letter-spacing: -0.03em;
}
.radar-tagline {
    font-size: 0.75rem !important;
    color: #3d5a7a !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.status-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #22c55e88;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Cards ─────────────────────────────────── */
.panel {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}
.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3d5a7a !important;
    text-transform: uppercase;
    letter-spacing: 0.25em;
    margin-bottom: 12px;
    border-bottom: 1px solid #1e2d3d;
    padding-bottom: 8px;
}

/* ── Evidence Cards ─────────────────────────── */
.evidence-card {
    background: #0a0f16;
    border: 1px solid #1e2d3d;
    border-left: 3px solid #4a9eff;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
    position: relative;
}
.evidence-card.visual { border-left-color: #a855f7; }
.evidence-card.semantic { border-left-color: #4a9eff; }
.ev-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.ev-filename {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #e6f0fb !important;
    font-weight: 600;
}
.ev-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
}
.ev-score.high { background: #14291a; color: #4ade80 !important; border: 1px solid #166534; }
.ev-score.med  { background: #1e1a05; color: #fbbf24 !important; border: 1px solid #854d0e; }
.ev-score.low  { background: #1a0f0f; color: #f87171 !important; border: 1px solid #7f1d1d; }
.ev-type-badge {
    font-size: 0.65rem !important;
    color: #3d5a7a !important;
    margin-bottom: 6px;
}
.ev-impression {
    font-size: 0.82rem !important;
    color: #6b7f97 !important;
    line-height: 1.5;
    font-style: italic;
}

/* ── Diagnostic Report ──────────────────────── */
.agent-block {
    background: #0a0f16;
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.agent-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d3d;
}
.agent-label.visual  { color: #a855f7 !important; }
.agent-label.corr    { color: #4a9eff !important; }
.agent-label.synth   { color: #22c55e !important; }
.agent-body { font-size: 0.88rem !important; color: #8b9bb4 !important; line-height: 1.7; }
.agent-body strong, .agent-body b { color: #c8d6e5 !important; }

/* ── Metrics Bar ────────────────────────────── */
.metrics-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    flex: 1;
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #4a9eff !important;
    display: block;
}
.metric-label { font-size: 0.68rem !important; color: #3d5a7a !important; text-transform: uppercase; letter-spacing: 0.15em; }

/* ── Streamlit Component Overrides ─────────────── */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #1e2d3d !important;
    border-radius: 8px !important;
}
textarea, input[type="text"] {
    background: #0a0f16 !important;
    border: 1px solid #1e2d3d !important;
    color: #c8d6e5 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stButton > button {
    background: #1a3a5c !important;
    color: #4a9eff !important;
    border: 1px solid #4a9eff !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #4a9eff !important;
    color: #080c10 !important;
}
.stButton > button[kind="primary"] {
    background: #4a9eff !important;
    color: #080c10 !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2563eb !important;
    border-color: #2563eb !important;
}

/* ── Chat ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #0d1117 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatMessage"] p { color: #c8d6e5 !important; font-size: 0.88rem !important; }
[data-testid="stChatInput"] textarea {
    background: #0d1117 !important;
    border: 1px solid #1e2d3d !important;
    color: #c8d6e5 !important;
}

/* ── Expander ─────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 8px !important;
}

/* ── Divider ──────────────────────────────────── */
hr { border-color: #1e2d3d !important; }

/* ── Spinner ──────────────────────────────────── */
.stSpinner > div { border-top-color: #4a9eff !important; }

/* ── Image caption ────────────────────────────── */
[data-testid="caption"] { color: #3d5a7a !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important; }

/* ── Sidebar nav ──────────────────────────────── */
.sidebar-section {
    background: #0a0f16;
    border: 1px solid #1e2d3d;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 12px;
}
.sidebar-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #3d5a7a !important;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 10px;
    display: block;
}

/* ── Image containers ─────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 6px;
    border: 1px solid #1e2d3d;
}

/* ── Scrollbar ────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #4a9eff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached resource loader
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising diagnostic models…")
def load_copilot():
    return DiagnosticCopilot()


# ─────────────────────────────────────────────
# Helper: render evidence card HTML
# ─────────────────────────────────────────────
def _score_class(score: float) -> str:
    if score >= 0.85: return "high"
    if score >= 0.65: return "med"
    return "low"

def render_evidence_card(case: dict, idx: int):
    match_type  = case.get("match_type", "")
    card_class  = "visual" if "Visual" in match_type else "semantic"
    score       = case.get("score", 0.0)
    score_cls   = _score_class(score)
    impression  = case.get("impression") or "Visual pattern match — no text impression available."
    xml_file    = case.get("xml_file", "unknown")

    st.markdown(f"""
    <div class="evidence-card {card_class}">
        <div class="ev-header">
            <span class="ev-filename">{xml_file}</span>
            <span class="ev-score {score_cls}">{score:.4f}</span>
        </div>
        <div class="ev-type-badge">{match_type}</div>
        <div class="ev-impression">{impression[:280]}{'…' if len(impression) > 280 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

    img_path = case.get("path")
    if img_path and os.path.exists(img_path):
        st.image(img_path, use_container_width=True)


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    copilot = load_copilot()

    # Session state init
    for key, default in [
        ("report", None),
        ("retrieved", []),
        ("heatmap_path", None),
        ("messages", []),
        ("last_image_name", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Header ──────────────────────────────────
    st.markdown("""
    <div class="radar-header">
        <span class="radar-logo">🔬 RADAR/AI</span>
        <div>
            <div class="radar-tagline">Multimodal Chest X-Ray Diagnostic Copilot</div>
            <div style="margin-top:4px">
                <span class="status-dot"></span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#22c55e !important;margin-left:6px;">
                    Actian VectorAI · CLIP ViT-B/32 · Gemini Multi-Agent
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.markdown('<span class="sidebar-section-label">⚙ System Config</span>', unsafe_allow_html=True)

        top_k = st.slider("Retrieval Depth (k)", 1, 6, 3, help="Number of similar cases to retrieve from Actian VectorAI")
        search_mode = st.radio(
            "Search Strategy",
            ["Hybrid (Recommended)", "Visual Only", "Semantic Only"],
            index=0,
        )

        st.markdown("---")
        st.markdown('<span class="sidebar-section-label">📡 Actian VectorAI</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-section">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#22c55e !important;">● ONLINE</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#3d5a7a !important;">localhost:50051</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="font-size:0.72rem;color:#3d5a7a !important;">Indexed Cases</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#4a9eff !important;">7,432</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;">
                <span style="font-size:0.72rem;color:#3d5a7a !important;">Collections</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#4a9eff !important;">cxr_text · cxr_images</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;">
                <span style="font-size:0.72rem;color:#3d5a7a !important;">Avg. Latency</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#4ade80 !important;">~38ms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("↺  Reset Session", use_container_width=True):
            for key in ["report", "retrieved", "heatmap_path", "messages", "last_image_name"]:
                st.session_state[key] = None if key != "retrieved" and key != "messages" else []
            st.rerun()

    # ── Input Column Layout ──────────────────────
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="panel-label">PATIENT INPUT</div>', unsafe_allow_html=True)

        uploaded_file   = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        clinical_notes  = st.text_area(
            "Clinical Observations",
            height=110,
            placeholder="e.g. 65M, chronic cough ×6 weeks, night sweats, 10 lb weight loss. Hx of smoking 30 pack-years.",
        )

        analyze_btn = st.button("⟶  ANALYZE & RETRIEVE EVIDENCE", type="primary", use_container_width=True)

        # Metrics
        if st.session_state.retrieved:
            n    = len(st.session_state.retrieved)
            best = max(c["score"] for c in st.session_state.retrieved)
            types = set(c.get("match_type", "") for c in st.session_state.retrieved)
            mode_str = "Hybrid" if len(types) > 1 else list(types)[0].replace("📝 ", "").replace("🖼️  ", "")
            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-box">
                    <span class="metric-value">{n}</span>
                    <span class="metric-label">Cases Retrieved</span>
                </div>
                <div class="metric-box">
                    <span class="metric-value">{best:.3f}</span>
                    <span class="metric-label">Best Match</span>
                </div>
                <div class="metric-box">
                    <span class="metric-value" style="font-size:0.9rem !important;">{mode_str}</span>
                    <span class="metric-label">Search Mode</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel-label">RADIOGRAPH VIEWER</div>', unsafe_allow_html=True)
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            if uploaded_file:
                st.image(uploaded_file, caption="INPUT · Patient Radiograph", use_container_width=True)
            else:
                st.markdown("""
                <div style="background:#0d1117;border:1px dashed #1e2d3d;border-radius:8px;
                            height:220px;display:flex;align-items:center;justify-content:center;">
                    <span style="color:#1e2d3d !important;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;">
                        AWAITING UPLOAD
                    </span>
                </div>
                """, unsafe_allow_html=True)
        with img_col2:
            if st.session_state.heatmap_path and os.path.exists(st.session_state.heatmap_path):
                st.image(st.session_state.heatmap_path, caption="AI ATTENTION · GradCAM Heatmap", use_container_width=True)
            else:
                st.markdown("""
                <div style="background:#0d1117;border:1px dashed #1e2d3d;border-radius:8px;
                            height:220px;display:flex;align-items:center;justify-content:center;">
                    <span style="color:#1e2d3d !important;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;">
                        HEATMAP PENDING
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # ── Analysis Logic ───────────────────────────
    if analyze_btn:
        if not uploaded_file and not clinical_notes.strip():
            st.error("Please provide a radiograph and/or clinical notes.")
        else:
            with st.spinner("Querying Actian evidence base and reasoning across agents…"):
                # Save temp image
                temp_path = None
                if uploaded_file:
                    temp_path = f"/tmp/radar_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.last_image_name = uploaded_file.name

                # Heatmap
                if temp_path:
                    hp = copilot.generate_heatmap(temp_path, text_query=clinical_notes or None)
                    if hp:
                        st.session_state.heatmap_path = hp
                        with col_right:
                            with img_col2:
                                st.image(hp, caption="AI ATTENTION · GradCAM Heatmap", use_container_width=True)

                # Retrieval
                use_text  = "Visual Only" not in search_mode
                use_image = "Semantic Only" not in search_mode
                retrieved = copilot.retrieve_similar_cases(
                    text_query  = clinical_notes if use_text else None,
                    image_path  = temp_path if use_image else None,
                    top_k       = top_k,
                )
                st.session_state.retrieved = retrieved

                # Multi-agent diagnosis
                report = copilot.generate_diagnosis(clinical_notes, temp_path, retrieved)
                st.session_state.report = report

                # Clear chat on new analysis
                st.session_state.messages = []

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            st.rerun()

    # ── Results Panel ────────────────────────────
    if st.session_state.report:
        st.markdown("---")
        report = st.session_state.report

        if report.get("error"):
            st.error(report["error"])
        else:
            res_left, res_right = st.columns([1.5, 1], gap="large")

            with res_left:
                # Agent 1: Visual Radiologist
                st.markdown("""
                <div class="agent-block">
                    <div class="agent-label visual">◈ Agent 01 · Visual Radiologist</div>
                    <div class="agent-body">
                """, unsafe_allow_html=True)
                st.markdown(report.get("visual", ""), unsafe_allow_html=False)
                st.markdown("</div></div>", unsafe_allow_html=True)

                # Agent 2: Clinical Integrator
                with st.expander("◈  Agent 02 · Clinical Integrator (Evidence Cross-Reference)", expanded=False):
                    st.markdown(
                        f'<div class="agent-body">{report.get("correlation", "")}</div>',
                        unsafe_allow_html=True
                    )

                # Agent 3: Final Synthesis
                st.markdown("""
                <div class="agent-block" style="border-left: 3px solid #22c55e;">
                    <div class="agent-label synth">◈ Agent 03 · Chief of Medicine · Final Synthesis</div>
                    <div class="agent-body">
                """, unsafe_allow_html=True)
                st.markdown(report.get("synthesis", ""), unsafe_allow_html=False)
                st.markdown("</div></div>", unsafe_allow_html=True)

                # ── Follow-up Chat ────────────────────────
                st.markdown("---")
                st.markdown('<div class="panel-label">💬 CLINICAL FOLLOW-UP</div>', unsafe_allow_html=True)

                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                if prompt := st.chat_input("Ask about the findings, differential, or next steps…"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Consulting…"):
                            # Build Gemini-compatible history
                            history = []
                            for m in st.session_state.messages[:-1]:
                                role  = "model" if m["role"] == "assistant" else "user"
                                history.append({"role": role, "parts": [m["content"]]})

                            reply = copilot.answer_followup(
                                question=prompt,
                                report=report,
                                chat_history=history,
                            )
                            st.markdown(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})

            with res_right:
                st.markdown('<div class="panel-label">📡 ACTIAN EVIDENCE BASE</div>', unsafe_allow_html=True)
                st.markdown(
                    '<p style="font-size:0.75rem !important;color:#3d5a7a !important;margin-bottom:14px;">'
                    'Mathematically similar cases from 7,432 indexed chest radiograph reports. '
                    'Retrieved via hybrid cosine similarity search.</p>',
                    unsafe_allow_html=True,
                )

                if st.session_state.retrieved:
                    for i, case in enumerate(st.session_state.retrieved):
                        render_evidence_card(case, i)
                else:
                    st.markdown('<p style="color:#3d5a7a !important;">No cases retrieved.</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()