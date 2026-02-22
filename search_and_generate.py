import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
load_dotenv()

import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from cortex import CortexClient

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL  = "ViT-B/32"
ACTIAN_ADDR = "localhost:50051"

# Ollama Config
OLLAMA_VISION_MODEL = "llava:13b" 
OLLAMA_TEXT_MODEL   = "mistral"

# Gemini Config
GEMINI_MODELS = ["gemini-2.5-flash"]

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER ABSTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class _GeminiProvider:
    def __init__(self):
        import google.generativeai as genai
        self._genai = genai
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key: raise RuntimeError("GEMINI_API_KEY missing.")
        genai.configure(api_key=api_key)
        self.model = self._connect()

    def _connect(self):
        for name in GEMINI_MODELS:
            try:
                m = self._genai.GenerativeModel(name)
                m.generate_content("ping", generation_config=self._genai.GenerationConfig(max_output_tokens=1))
                return m
            except: continue
        raise RuntimeError("Gemini connection failed.")

    def vision_text(self, prompt, image): return self.model.generate_content([prompt, image]).text
    def text(self, prompt): return self.model.generate_content(prompt).text
    def chat(self, messages):
        history = messages[:-1]
        last = messages[-1]["parts"][0]
        return self.model.start_chat(history=history).send_message(last).text

class _OllamaProvider:
    def __init__(self):
        import ollama
        self._ollama = ollama
        try:
            self._ollama.list()
            print(f"[Ollama] ✅ Connected")
        except:
            print("[Ollama] ⚠️ Connection failed. Is Ollama running?")

    def vision_text(self, prompt, image):
        import io
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        try:
            r = self._ollama.generate(model=OLLAMA_VISION_MODEL, prompt=prompt, images=[buf.getvalue()])
            return r['response']
        except Exception as e:
            return f"Ollama Error: {e}"

    def text(self, prompt):
        try:
            r = self._ollama.generate(model=OLLAMA_TEXT_MODEL, prompt=prompt)
            return r['response']
        except Exception as e:
            return f"Ollama Error: {e}"

    def chat(self, messages):
        ollama_msgs = []
        for m in messages:
            role = "assistant" if m["role"] == "model" else "user"
            ollama_msgs.append({"role": role, "content": m["parts"][0]})
        try:
            r = self._ollama.chat(model=OLLAMA_TEXT_MODEL, messages=ollama_msgs)
            return r['message']['content']
        except Exception as e:
            return f"Ollama Error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticCopilot:
    def __init__(self, provider="gemini"):
        # Force reload environment variables to pick up key changes
        load_dotenv(override=True)
        
        print(f"[Init] Loading {EMBED_MODEL} & {CLIP_MODEL} on {DEVICE}...")
        self.text_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.clip_model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
        self.clip_model.eval()
        
        self.provider_name = provider
        self.llm = self._init_llm(provider)

    def _init_llm(self, provider):
        try:
            print(f"[Init] Connecting to {provider.upper()}...")
            if provider == "ollama": return _OllamaProvider()
            return _GeminiProvider()
        except Exception as e:
            print(f"LLM Provider Error: {e}")
            return None

    def retrieve_similar_cases(self, text_query=None, image_path=None, top_k=3):
        results = []
        try:
            with CortexClient(ACTIAN_ADDR) as client:
                all_cols = [c.name for c in client.list_collections()]
                text_col = "med_text" if "med_text" in all_cols else "cxr_text"
                img_col = "med_images" if "med_images" in all_cols else "cxr_images"

                if text_query and text_query.strip():
                    vec = self.text_model.encode([text_query])[0].tolist()
                    for hit in client.search(text_col, vec, top_k=top_k):
                        _, payload = client.get(text_col, hit.id)
                        results.append({"match_type": "📝 Semantic", "score": hit.score, "xml_file": payload.get("xml_file", "ID"), "impression": payload.get("findings", payload.get("impression", ""))})
                
                if image_path and os.path.exists(image_path):
                    img = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        feats = self.clip_model.encode_image(img)
                        vec = (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy().flatten().tolist()
                    for hit in client.search(img_col, vec, top_k=top_k):
                        _, payload = client.get(img_col, hit.id)
                        results.append({"match_type": "🖼️  Visual", "score": hit.score, "xml_file": payload.get("xml_file", "ID"), "impression": payload.get("findings", payload.get("impression", "")), "path": payload.get("path")})

            results.sort(key=lambda x: x["score"], reverse=True)
            seen, deduped = set(), []
            for r in results:
                if r["xml_file"] not in seen:
                    seen.add(r["xml_file"]); deduped.append(r)
            return deduped[:top_k]
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return []

    def generate_diagnosis(self, clinical_notes, image_path, retrieved_cases, domain="Radiologist"):
        if not self.llm: return {"error": f"LLM Provider ({self.provider_name}) Unavailable."}

        evidence = "\n".join(f"[{i+1}] {c['xml_file']} (score {c['score']:.4f}): {c['impression']}" for i, c in enumerate(retrieved_cases))

        # Agent 1: Visual Analyst
        visual_findings = "No image provided."
        if image_path:
            prompt_v = f"Act as an expert {domain}. Provide a structured objective technical analysis of this medical image. Cover all relevant landmarks and incidental findings."
            try: visual_findings = self.llm.vision_text(prompt_v, Image.open(image_path))
            except Exception as e: visual_findings = f"Error: {e}"

        # Agent 2: Clinical Integrator
        prompt_i = f"Act as a {domain}. Integrate Visual Findings: {visual_findings} with Patient Notes: {clinical_notes} and Actian Evidence Base: {evidence}. Identify convergent patterns and differential diagnoses."
        try: correlation = self.llm.text(prompt_i)
        except Exception as e: correlation = f"Error: {e}"

        # Agent 3: Synthesis
        prompt_s = f"Act as Chief of Medicine. Finalize this diagnostic report based on: {correlation}. Use format: Primary Diagnosis, Confidence %, Evidence Basis (citing cases), and Recommendations. Professional markdown."
        try: synthesis = self.llm.text(prompt_s)
        except Exception as e: synthesis = f"Error: {e}"

        return {"visual": visual_findings, "correlation": correlation, "synthesis": synthesis, "error": None}

    def answer_followup(self, question, report, chat_history):
        if not self.llm: return "LLM unavailable."
        ctx = f"VISUAL: {report.get('visual','')}\nCORR: {report.get('correlation','')}\nSYNTH: {report.get('synthesis','')}"
        msgs = [{"role": "user", "parts": [f"You are a clinical decision assistant. Report: {ctx}. User: {question}"]}]
        try: return self.llm.chat(msgs)
        except Exception as e: return f"Error: {e}"
