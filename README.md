# 🔬 RadarAI: Universal Multimodal Diagnostic Copilot

**RadarAI** is a domain-agnostic, retrieval-augmented generation (RAG) system designed to assist medical professionals. It combines the high-performance hybrid search of **Actian VectorAI** with the multimodal reasoning of **Google Gemini** (or local **Ollama** models) to provide evidence-based diagnostic insights.

Unlike standard "black box" AI, RadarAI grounds every diagnosis in retrieved historical evidence, displaying similar verified cases alongside its analysis to ensure clinical traceability.

---

## 🌟 Key Features

*   **🧠 Multi-Agent "Factory" Architecture**
    *   **Agent 1 (Visual Analyst):** specialized vision model (Gemini 1.5 Flash / Llava) extracts objective technical findings.
    *   **Agent 2 (Clinical Integrator):** correlates visual findings with patient history and retrieved evidence.
    *   **Agent 3 (Chief Synthesis):** generates the final professional medical report with confidence scores.
*   **⚡ Powered by Actian VectorAI**
    *   Leverages Actian's hybrid engine for sub-50ms retrieval of high-dimensional multimodal vectors.
    *   Uses **CLIP (ViT-B/32)** for visual search and **all-MiniLM-L6-v2** for semantic text search.
*   **🔌 Domain-Agnostic Design**
    *   Not limited to X-rays. The architecture supports **Dermatology**, **Pathology**, **MRI**, and **CT** scans.
    *   Includes universal ingestors for any image+text medical dataset.
*   **☁️ Hybrid AI Backend**
    *   **Cloud Mode:** Uses Google **Gemini 1.5 Pro/Flash** for state-of-the-art reasoning.
    *   **Local Mode:** Fully offline capable using **Ollama** (Llava + Mistral) for privacy-first deployments.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   **Docker Desktop** (for the database)
*   **Python 3.9+**
*   **Google AI Studio API Key** (optional, for Cloud mode)
*   **Ollama** (optional, for Local mode)

### 2. Start the Database
Spin up the Actian VectorAI instance using Docker:
```bash
docker run -d \
  --name actian-vectorai \
  -p 50051:50051 \
  williamimoh/actian-vectorai-db:1.0b
```

### 3. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Actian Client (Critical step)
# Ensure you have the wheel file or install from the repo
pip install actiancortex-0.1.0b1-py3-none-any.whl

# Install project requirements
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the root directory:
```bash
GEMINI_API_KEY=your_google_api_key_here
LLM_PROVIDER=gemini  # or 'ollama'
```

---

## 📊 Data Ingestion (Build Your Evidence Base)

RadarAI needs verified medical data to function. We provide scripts for two major datasets:

### Option A: Dermatology (HAM10000)
```bash
# 1. Download Data
mkdir -p data/ham10000
# (Manually download or use provided curl commands in dev docs)

# 2. Ingest & Create Synthetic Notes
python ingest_ham10000.py

# 3. Index into Actian
python index_data.py
```

### Option B: Chest X-Rays (OpenI)
```bash
# 1. Download Data
./download_data.sh

# 2. Parse XML Reports
python parse_data.py

# 3. Index into Actian
python index_data.py
```

---

## 🚀 Usage

Launch the professional dashboard:
```bash
streamlit run app.py
```

### How to Demo:
1.  **Select Expert Persona:** Choose "Radiologist" or "Dermatologist" from the sidebar.
2.  **Switch Provider:** Toggle between **Gemini** (Cloud) and **Ollama** (Local).
3.  **Upload:** Drop a medical image (X-ray or Skin Lesion).
4.  **Observe:** Watch the "Agent Factory" execute sequentially:
    *   Retrieving similar historical cases from Actian.
    *   Generating a visual analysis.
    *   Synthesizing a final report with cited evidence.

---

## 📂 Project Structure

*   `app.py`: Main Streamlit dashboard with "Medical-Grade" UI.
*   `search_and_generate.py`: Core logic for the RAG pipeline, LLM abstraction, and Multi-Agent factory.
*   `index_data.py`: Handles vectorization (CLIP/MiniLM) and batch upserting to Actian VectorAI.
*   `ingest_*.py`: Specialized ETL scripts for transforming raw medical datasets into the `processed_data.csv` standard.
*   `validate_system.py`: Mathematical sanity check to verify vector retrieval accuracy.

---

## 🏆 Hacklytics 2026 Submission
Built for the **Healthcare** and **Best Use of Actian VectorAI DB** tracks.

**Technology Stack:**
*   **Database:** Actian VectorAI
*   **LLMs:** Google Gemini, Ollama (Mistral/Llava)
*   **Embeddings:** OpenAI CLIP, Sentence-Transformers
*   **Frontend:** Streamlit
