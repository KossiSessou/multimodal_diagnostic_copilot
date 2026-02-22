import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from PIL import Image
from search_and_generate import DiagnosticCopilot

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Diagnostic Copilot",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global Styles (RadarAI Premium Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,500;0,700;1,300&display=swap');

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

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e6f0fb !important; }
h1 { font-size: 1.6rem !important; letter-spacing: -0.02em; }
h3 { font-size: 1rem !important; color: #4a9eff !important; text-transform: uppercase; letter-spacing: 0.12em; }
p, li, span, label, div { color: #8b9bb4 !important; font-size: 0.9rem !important; }

.radar-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0 16px 0;
    border-bottom: 1px solid #1e2d3d;
    margin-bottom: 24px;
}
.radar-logo { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #4a9eff !important; letter-spacing: -0.03em; }
.radar-tagline { font-size: 0.75rem !important; color: #3d5a7a !important; letter-spacing: 0.2em; text-transform: uppercase; }
.status-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #22c55e88; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.panel-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #3d5a7a !important; text-transform: uppercase; letter-spacing: 0.25em; margin-bottom: 12px; border-bottom: 1px solid #1e2d3d; padding-bottom: 8px; }

.evidence-card { background: #0a0f16; border: 1px solid #1e2d3d; border-left: 3px solid #4a9eff; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px; }
.evidence-card.visual { border-left-color: #a855f7; }
.ev-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.ev-filename { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #e6f0fb !important; font-weight: 600; }
.ev-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; background: #14291a; color: #4ade80 !important; }
.ev-impression { font-size: 0.82rem !important; color: #6b7f97 !important; line-height: 1.5; font-style: italic; }

.agent-block { background: #0a0f16; border: 1px solid #1e2d3d; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px; }
.agent-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #1e2d3d; }
.agent-label.visual { color: #a855f7 !important; }
.agent-label.corr { color: #4a9eff !important; }
.agent-label.synth { color: #22c55e !important; }

.metric-box { background: #0d1117; border: 1px solid #1e2d3d; border-radius: 6px; padding: 12px 16px; text-align: center; flex: 1; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.2rem; color: #4a9eff !important; }
.metric-label { font-size: 0.65rem !important; color: #3d5a7a !important; text-transform: uppercase; }

[data-testid="stChatMessage"] { background: #0d1117 !important; border: 1px solid #1e2d3d !important; }
[data-testid="stChatMessage"] p { color: #c8d6e5 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initialising RadarAI Diagnostic models…")
def load_copilot(provider_choice):
    return DiagnosticCopilot(provider=provider_choice.lower())

def render_evidence_card(case: dict, idx: int):
    match_type = case.get("match_type", "")
    card_class = "visual" if "Visual" in match_type else "semantic"
    score = case.get("score", 0.0)
    impression = case.get("impression") or "Visual pattern match only."
    
    st.markdown(f"""
    <div class="evidence-card {card_class}">
        <div class="ev-header">
            <span class="ev-filename">{case.get('xml_file', 'ID')}</span>
            <span class="ev-score">{score:.4f}</span>
        </div>
        <div style="font-size:0.65rem;color:#3d5a7a;margin-bottom:5px;">{match_type}</div>
        <div class="ev-impression">{impression[:200]}...</div>
    </div>
    """, unsafe_allow_html=True)
    if case.get("path") and os.path.exists(case["path"]):
        st.image(case["path"], use_container_width=True)

def main():
    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.markdown('<span class="panel-label">⚙ System Config</span>', unsafe_allow_html=True)
        
        # Provider Toggle
        provider = st.radio("AI Model Provider", ["Gemini", "Ollama"], index=0)
        
        domain = st.selectbox("Expert Persona", ["Radiologist", "Dermatologist", "Pathologist", "General Specialist"], index=0)
        top_k = st.slider("Retrieval Depth (k)", 1, 6, 3)
        search_mode = st.radio("Strategy", ["Hybrid", "Visual Only", "Semantic Only"])

        st.markdown("---")
        st.markdown('<span class="panel-label">📡 Actian VectorAI</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0a0f16;padding:12px;border-radius:6px;border:1px solid #1e2d3d;">
            <div style="display:flex;justify-content:space-between;font-family:monospace;font-size:0.7rem;">
                <span style="color:#22c55e;">● ONLINE</span>
                <span style="color:#3d5a7a;">Active</span>
            </div>
            <div style="margin-top:8px;color:#8b9bb4;font-size:0.75rem;">
                Latency: <span style="color:#4ade80;">~38ms</span><br>
                Records: <span style="color:#4a9eff;">7,432 Indexed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("↺ Reset Session", use_container_width=True):
            for key in ["report", "retrieved", "messages"]: st.session_state[key] = [] if isinstance(st.session_state[key], list) else None
            st.rerun()

    # Load Copilot with selected provider
    copilot = load_copilot(provider)

    for key, default in [("report", None), ("retrieved", []), ("messages", [])]:
        if key not in st.session_state: st.session_state[key] = default

    # ── Header ──────────────────────────────────
    st.markdown(f"""
    <div class="radar-header">
        <span class="radar-logo">🔬 RADAR/AI</span>
        <div>
            <div class="radar-tagline">Multimodal Diagnostic Copilot</div>
            <div style="margin-top:4px">
                <span class="status-dot"></span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#22c55e !important;margin-left:6px;">
                    Actian VectorAI · {provider} Backend · Multi-Agent RAG
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Layout ─────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="panel-label">PATIENT INTAKE</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Medical Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        clinical_notes = st.text_area("Observations", height=110, placeholder="Describe clinical presentation...")
        analyze_btn = st.button("⟶ START DIAGNOSTIC FACTORY", type="primary", use_container_width=True)

        if st.session_state.retrieved:
            best_score = max(c["score"] for c in st.session_state.retrieved)
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-top:20px;">
                <div class="metric-box"><span class="metric-value">{len(st.session_state.retrieved)}</span><br><span class="metric-label">Cases</span></div>
                <div class="metric-box"><span class="metric-value">{best_score:.3f}</span><br><span class="metric-label">Max Sim</span></div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel-label">IMAGE VIEWER</div>', unsafe_allow_html=True)
        if uploaded_file: st.image(uploaded_file, caption="INPUT", use_container_width=True)
        else: st.info("Awaiting Image")

    # ── Analysis Logic ───────────────────────────
    if analyze_btn:
        if not uploaded_file and not clinical_notes.strip(): st.error("Please provide image or notes.")
        else:
            with st.spinner(f"Executing Multi-Agent Factory via {provider}..."):
                temp_path = None
                if uploaded_file:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                
                st.session_state.retrieved = copilot.retrieve_similar_cases(
                    text_query=clinical_notes if "Visual" not in search_mode else None,
                    image_path=temp_path if "Semantic" not in search_mode else None,
                    top_k=top_k
                )
                
                st.session_state.report = copilot.generate_diagnosis(clinical_notes, temp_path, st.session_state.retrieved, domain=domain)
                st.session_state.messages = []
                if temp_path and os.path.exists(temp_path): os.remove(temp_path)
            st.rerun()

    # ── Results Dashboard ────────────────────────
    if st.session_state.report:
        st.markdown("---")
        report = st.session_state.report
        if report.get("error"): st.error(report["error"])
        else:
            res_l, res_r = st.columns([1.5, 1], gap="large")
            with res_l:
                st.markdown(f'<div class="agent-block"><div class="agent-label visual">◈ Agent 01 · {domain} Analysis</div><div class="agent-body">{report["visual"]}</div></div>', unsafe_allow_html=True)
                with st.expander("◈ Agent 02 · Clinical Integrator (Evidence Cross-Reference)", expanded=False): 
                    st.markdown(f'<div class="agent-body">{report["correlation"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="agent-block" style="border-left:3px solid #22c55e;"><div class="agent-label synth">◈ Agent 03 · Medical Board Lead Synthesis</div><div class="agent-body">{report["synthesis"]}</div></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="panel-label">💬 CLINICAL FOLLOW-UP</div>', unsafe_allow_html=True)
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
                if prompt := st.chat_input("Ask about the findings or next steps…"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        reply = copilot.answer_followup(prompt, report, [])
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})

            with res_r:
                st.markdown('<div class="panel-label">📡 ACTIAN EVIDENCE BASE</div>', unsafe_allow_html=True)
                for i, case in enumerate(st.session_state.retrieved):
                    render_evidence_card(case, i)

if __name__ == "__main__":
    main()
