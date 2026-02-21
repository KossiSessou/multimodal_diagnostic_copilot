# import os
# # Force pure-python implementation to bypass version conflicts
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# from dotenv import load_dotenv
# load_dotenv() # Load variables from .env file

# import torch
# import clip
# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from cortex import CortexClient
# import google.generativeai as genai
# from pytorch_grad_cam import GradCAM, EigenCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import cv2
# import numpy as np

# # Configuration
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
# CLIP_MODEL_NAME = "ViT-B/32"
# ACTIAN_ADDR = "localhost:50051"

# class DiagnosticCopilot:
#     def __init__(self):
#         print(f"Loading models on {DEVICE}...")
#         self.text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
#         self.clip_model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
#         self.genai_model = self._initialize_gemini()

#     def _initialize_gemini(self):
#         api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             print("❌ GEMINI_API_KEY not found in environment.")
#             return None
            
#         genai.configure(api_key=api_key)
        
#         # Expanded list of model names to avoid 404s
#         model_names = [
#             'gemini-2.5-flash',
#             'gemini-1.5-flash', 
#             'gemini-1.5-flash-latest', 
#             'gemini-2.0-flash-exp', 
#             'gemini-1.5-pro',
#             'gemini-pro'
#         ]
        
#         for name in model_names:
#             try:
#                 model = genai.GenerativeModel(name)
#                 # Just a very basic check
#                 model.generate_content("ping", generation_config={"max_output_tokens": 1})
#                 print(f"✅ Successfully initialized Gemini with model: {name}")
#                 return model
#             except Exception as e:
#                 print(f"⚠️ Failed to initialize {name}: {str(e)}")
#                 continue
        
#         print("❌ All Gemini model variants failed to initialize.")
#         return None

#     def generate_heatmap(self, image_path, text_query=None):
#         try:
#             raw_image = Image.open(image_path).convert('RGB')
#             img_np = np.array(raw_image.resize((224, 224))) / 255.0
#             input_tensor = self.preprocess(raw_image).unsqueeze(0).to(DEVICE)

#             def reshape_transform(tensor):
#                 result = tensor[1:, :, :].reshape(7, 7, tensor.size(1), tensor.size(2))
#                 result = result.permute(2, 3, 0, 1) 
#                 return result

#             target_layers = [self.clip_model.visual.transformer.resblocks[-1]]
#             grayscale_cam = None

#             if text_query and text_query != "All Cases":
#                 text_tokens = clip.tokenize([text_query]).to(DEVICE)
#                 with torch.no_grad():
#                     text_embedding = self.clip_model.encode_text(text_tokens).float()
#                     text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

#                 class SemanticTarget:
#                     def __init__(self, text_emb): self.text_emb = text_emb
#                     def __call__(self, model_output):
#                         return torch.matmul(model_output, self.text_emb.t())

#                 cam = GradCAM(model=self.clip_model.visual, target_layers=target_layers, reshape_transform=reshape_transform)
#                 grayscale_cam = cam(input_tensor=input_tensor, targets=[SemanticTarget(text_embedding)])[0]
#             else:
#                 cam = EigenCAM(model=self.clip_model.visual, target_layers=target_layers, reshape_transform=reshape_transform)
#                 grayscale_cam = cam(input_tensor=input_tensor)[0]
            
#             grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
#             visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
#             heatmap_path = os.path.join(os.getcwd(), f"heatmap_{os.path.basename(image_path)}")
#             cv2.imwrite(heatmap_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
#             return heatmap_path
#         except Exception as e:
#             print(f"Heatmap Error: {e}")
#             return None

#     def retrieve_similar_cases(self, text_query=None, image_path=None, top_k=3):
#         results = []
#         with CortexClient(ACTIAN_ADDR) as client:
#             if text_query:
#                 query_vec = self.text_model.encode([text_query])[0].tolist()
#                 text_hits = client.search("cxr_text", query_vec, top_k=top_k)
#                 for hit in text_hits:
#                     _, payload = client.get("cxr_text", hit.id)
#                     results.append({
#                         "type": "Semantic Match",
#                         "score": hit.score,
#                         "xml_file": payload.get("xml_file"),
#                         "impression": payload.get("impression")
#                     })
#             if image_path:
#                 image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
#                 with torch.no_grad():
#                     image_features = self.clip_model.encode_image(image)
#                     image_features /= image_features.norm(dim=-1, keepdim=True)
#                     query_vec = image_features.cpu().numpy().flatten().tolist()
#                 image_hits = client.search("cxr_images", query_vec, top_k=top_k)
#                 for hit in image_hits:
#                     _, payload = client.get("cxr_images", hit.id)
#                     results.append({
#                         "type": "Visual Match",
#                         "score": hit.score,
#                         "xml_file": payload.get("xml_file"),
#                         "path": payload.get("path")
#                     })
#         results = sorted(results, key=lambda x: x['score'], reverse=True)
#         return results[:top_k]

#     def generate_diagnosis(self, query_text, query_image_path, retrieved_cases):
#         # Retry initialization if it failed at startup
#         if not self.genai_model:
#             self.genai_model = self._initialize_gemini()
            
#         if not self.genai_model:
#             api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#             if not api_key:
#                 return "⚠️ **API Key Missing**: Please set `GEMINI_API_KEY` in your `.env` file or environment."
#             return "❌ **Gemini Error**: Could not connect to any Gemini models. Check your API key permissions in Google AI Studio."

#         context_data = ""
#         for i, case in enumerate(retrieved_cases):
#             context_data += f"[Evidence {i+1} | Source: {case['xml_file']}]: {case.get('impression', 'Visual match only')}\n"

#         # Agent 1: Visual Radiologist
#         visual_findings = "No image provided."
#         if query_image_path and os.path.exists(query_image_path):
#             try:
#                 img = Image.open(query_image_path)
#                 visual_findings = self.genai_model.generate_content(["ACT AS: Specialist Radiologist. Analyze this X-ray and list objective technical findings.", img]).text
#             except Exception as e:
#                 visual_findings = f"Visual analysis failed: {str(e)}"

#         # Agent 2: Clinical Integration & Evidence Synthesis
#         correlation_prompt = f"""
#         ACT AS: Diagnostic Specialist.
#         INTEGRATE: Visual Findings ({visual_findings}), Patient Notes ({query_text}), and Actian Evidence Base below.
        
#         EVIDENCE BASE (Retrieved from Actian VectorDB):
#         {context_data}
        
#         TASK: Correlate the new patient's visuals with the historical gold-standard cases. 
#         Identify the most likely diagnosis based on this cross-reference.
#         """
#         try:
#             correlation = self.genai_model.generate_content(correlation_prompt).text
#         except Exception as e:
#             correlation = f"Correlation failed: {str(e)}"

#         # Agent 3: Final Clinical Impression
#         final_prompt = f"""
#         ACT AS: Chief of Medicine. Finalize the diagnostic report.
#         CONTEXT: {correlation}
        
#         FORMAT:
#         ### 🩺 FINAL CLINICAL IMPRESSION
#         **Diagnosis:** [Concise diagnosis]
#         **Certainty:** [Confidence %]
#         **Rationale:** [One sentence explicitly citing the Evidence Base]
#         **Recommendations:** [Actionable steps]
#         """
#         try:
#             synthesis = self.genai_model.generate_content(final_prompt).text
#         except Exception as e:
#             synthesis = f"Synthesis failed: {str(e)}"

#         return f"### 👁️ Radiologist Analysis\n{visual_findings}\n\n---\n### 🔬 Clinical Correlation\n{correlation}\n\n---\n{synthesis}"



# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# from dotenv import load_dotenv
# load_dotenv()

# import torch
# import clip
# import numpy as np
# import cv2
# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from cortex import CortexClient
# import google.generativeai as genai
# from pytorch_grad_cam import GradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # ─────────────────────────────────────────────
# # Configuration
# # ─────────────────────────────────────────────
# DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# TEXT_MODEL    = "all-MiniLM-L6-v2"
# CLIP_MODEL    = "ViT-B/32"
# ACTIAN_ADDR   = "localhost:50051"

# # Gemini model priority list (most capable first)
# GEMINI_MODELS = [
#     "gemini-3-flash",
#     "gemini-2.0-flash-exp",
#     "gemini-1.5-flash-latest",
#     "gemini-1.5-flash",
#     "gemini-1.5-pro",
# ]


# class DiagnosticCopilot:
#     """
#     Multimodal RAG pipeline for chest X-ray clinical decision support.
    
#     Architecture:
#         1. CLIP ViT-B/32   → visual embeddings + GradCAM/EigenCAM heatmaps
#         2. all-MiniLM-L6   → text embeddings
#         3. Actian VectorAI → hybrid vector search (cosine)
#         4. Gemini (multi-agent) → visual analysis → clinical correlation → synthesis
#     """

#     def __init__(self):
#         print(f"[Init] Loading embedding models on {DEVICE}…")
#         self.text_model  = SentenceTransformer(TEXT_MODEL, device=DEVICE)
#         self.clip_model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
#         self.clip_model.eval()
#         self.gemini      = self._init_gemini()
#         print("[Init] DiagnosticCopilot ready.")

#     # ──────────────────────────────────────────
#     # Gemini initialisation (fail-fast, no I/O per attempt)
#     # ──────────────────────────────────────────
#     def _init_gemini(self):
#         api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             print("[Gemini] ❌  GEMINI_API_KEY not found.")
#             return None

#         genai.configure(api_key=api_key)

#         for name in GEMINI_MODELS:
#             try:
#                 model = genai.GenerativeModel(name)
#                 # Lightweight probe — avoids wasting quota on repeated "ping" calls
#                 model.generate_content(
#                     "Hi",
#                     generation_config=genai.GenerationConfig(max_output_tokens=1),
#                 )
#                 print(f"[Gemini] ✅  Connected → {name}")
#                 return model
#             except Exception as err:
#                 print(f"[Gemini] ⚠️  {name} unavailable: {err}")

#         print("[Gemini] ❌  All model variants failed.")
#         return None

#     # ──────────────────────────────────────────
#     # Explainable AI: Attention Heatmap
#     # ──────────────────────────────────────────
#     def generate_heatmap(self, image_path: str, text_query: str | None = None) -> str | None:
#         """
#         Returns path to saved heatmap PNG.
#         - text_query provided → Text-Guided Grad-CAM (semantic attention)
#         - text_query absent   → EigenCAM (structural saliency)
#         """
#         try:
#             raw  = Image.open(image_path).convert("RGB")
#             img_np = np.array(raw.resize((224, 224))) / 255.0          # float32 [0,1]
#             tensor = self.preprocess(raw).unsqueeze(0).to(DEVICE)

#             # ViT reshape: [Tokens, B, C] → [B, C, H, W]
#             def reshape_transform(t):
#                 h = t[1:].reshape(7, 7, t.size(1), t.size(2))
#                 return h.permute(2, 3, 0, 1)

#             target_layers = [self.clip_model.visual.transformer.resblocks[-1]]

#             if text_query and text_query.strip():
#                 # ── Semantic / Text-Guided Grad-CAM ──────────────────────────
#                 tokens = clip.tokenize([text_query]).to(DEVICE)
#                 with torch.no_grad():
#                     t_emb = self.clip_model.encode_text(tokens).float()
#                     t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)

#                 class _SemanticTarget:
#                     def __init__(self, emb): self.emb = emb
#                     def __call__(self, out):
#                         # Cosine similarity as scalar target
#                         return (out / (out.norm(dim=-1, keepdim=True) + 1e-8)) @ self.emb.t()

#                 cam = GradCAM(
#                     model=self.clip_model.visual,
#                     target_layers=target_layers,
#                     reshape_transform=reshape_transform,
#                 )
#                 grayscale = cam(input_tensor=tensor, targets=[_SemanticTarget(t_emb)])[0]
#                 caption   = f"Semantic attention: '{text_query}'"

#             else:
#                 # ── Structural / EigenCAM ─────────────────────────────────────
#                 cam = EigenCAM(
#                     model=self.clip_model.visual,
#                     target_layers=target_layers,
#                     reshape_transform=reshape_transform,
#                 )
#                 grayscale = cam(input_tensor=tensor)[0]
#                 caption   = "Structural saliency (EigenCAM)"

#             # Normalise + overlay
#             lo, hi = grayscale.min(), grayscale.max()
#             grayscale = (grayscale - lo) / (hi - lo + 1e-8)
#             vis = show_cam_on_image(img_np.astype(np.float32), grayscale, use_rgb=True)

#             out_path = os.path.join(os.getcwd(), f"heatmap_{os.path.basename(image_path)}")
#             cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
#             print(f"[Heatmap] Saved → {out_path}  ({caption})")
#             return out_path

#         except Exception as e:
#             print(f"[Heatmap] Error: {e}")
#             return None

#     # ──────────────────────────────────────────
#     # Hybrid Retrieval (Text + Visual)
#     # ──────────────────────────────────────────
#     def retrieve_similar_cases(
#         self,
#         text_query: str | None = None,
#         image_path: str | None = None,
#         top_k: int = 3,
#     ) -> list[dict]:
#         """
#         Hybrid retrieval from Actian VectorAI:
#           - Semantic text search  → cxr_text   (MiniLM, dim=384)
#           - Visual image search   → cxr_images (CLIP,   dim=512)
#         Results are score-sorted and deduplicated by xml_file.
#         """
#         results = []

#         with CortexClient(ACTIAN_ADDR) as client:

#             # 1. Text branch
#             if text_query and text_query.strip():
#                 vec = self.text_model.encode([text_query])[0].tolist()
#                 for hit in client.search("cxr_text", vec, top_k=top_k):
#                     _, payload = client.get("cxr_text", hit.id)
#                     results.append({
#                         "match_type": "📝 Semantic",
#                         "score":      hit.score,
#                         "xml_file":   payload.get("xml_file", "unknown"),
#                         "impression": payload.get("impression", ""),
#                         "path":       None,
#                     })

#             # 2. Visual branch
#             if image_path and os.path.exists(image_path):
#                 img = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
#                 with torch.no_grad():
#                     feats = self.clip_model.encode_image(img)
#                     feats = feats / feats.norm(dim=-1, keepdim=True)
#                     vec   = feats.cpu().numpy().flatten().tolist()

#                 for hit in client.search("cxr_images", vec, top_k=top_k):
#                     _, payload = client.get("cxr_images", hit.id)
#                     results.append({
#                         "match_type": "🖼️  Visual",
#                         "score":      hit.score,
#                         "xml_file":   payload.get("xml_file", "unknown"),
#                         "impression": payload.get("impression", ""),
#                         "path":       payload.get("path"),
#                     })

#         # Sort by score, deduplicate by xml_file (keep highest score)
#         results.sort(key=lambda x: x["score"], reverse=True)
#         seen, deduped = set(), []
#         for r in results:
#             if r["xml_file"] not in seen:
#                 seen.add(r["xml_file"])
#                 deduped.append(r)

#         return deduped[:top_k]

#     # ──────────────────────────────────────────
#     # Multi-Agent Diagnostic Pipeline
#     # ──────────────────────────────────────────
#     def generate_diagnosis(
#         self,
#         clinical_notes: str,
#         image_path:     str | None,
#         retrieved_cases: list[dict],
#     ) -> dict:
#         """
#         Three-agent pipeline:
#           Agent 1 → Visual Radiologist   (image-grounded)
#           Agent 2 → Clinical Integrator  (notes + evidence)
#           Agent 3 → Chief Synthesis      (final impression)

#         Returns a dict with keys: visual, correlation, synthesis, error
#         """
#         if not self.gemini:
#             self.gemini = self._init_gemini()
#         if not self.gemini:
#             return {
#                 "error": (
#                     "⚠️ **Gemini Unavailable** — Set `GEMINI_API_KEY` in your `.env` file "
#                     "and restart the app."
#                 )
#             }

#         # Build evidence block
#         evidence_block = "\n".join(
#             f"  [{i+1}] {c['xml_file']} (score {c['score']:.4f}): "
#             f"{c.get('impression') or 'visual pattern match only'}"
#             for i, c in enumerate(retrieved_cases)
#         ) or "  No cases retrieved."

#         # ── Agent 1: Visual Radiologist ───────────────────────────────────────
#         visual_findings = "No radiograph provided."
#         if image_path and os.path.exists(image_path):
#             try:
#                 img = Image.open(image_path)
#                 prompt_visual = (
#                     "You are an expert chest radiologist. "
#                     "Provide a structured, objective technical analysis of this X-ray. "
#                     "Cover: cardiac silhouette, pulmonary vascularity, lung parenchyma, "
#                     "pleural spaces, bony structures, mediastinum, and any incidental findings. "
#                     "Be precise and avoid speculation."
#                 )
#                 visual_findings = self.gemini.generate_content([prompt_visual, img]).text
#             except Exception as e:
#                 visual_findings = f"Visual analysis error: {e}"

#         # ── Agent 2: Clinical Integrator ─────────────────────────────────────
#         prompt_integrate = f"""You are a Clinical Integration Specialist.

# VISUAL FINDINGS (from Radiologist Agent):
# {visual_findings}

# PATIENT CLINICAL NOTES:
# {clinical_notes or "None provided."}

# ACTIAN VECTORAI EVIDENCE BASE (top similar historical cases):
# {evidence_block}

# TASK:
# Cross-reference the visual findings and clinical notes against the retrieved historical cases.
# Identify convergent diagnostic patterns. Note any discrepancies or atypical features.
# Generate a ranked differential diagnosis with supporting evidence citations (e.g., "[Case 2]").
# Keep the tone clinical and concise."""

#         try:
#             correlation = self.gemini.generate_content(prompt_integrate).text
#         except Exception as e:
#             correlation = f"Clinical correlation error: {e}"

#         # ── Agent 3: Chief of Medicine Synthesis ──────────────────────────────
#         prompt_synthesis = f"""You are the Chief of Medicine conducting final case review.

# CLINICAL CORRELATION ANALYSIS:
# {correlation}

# Generate a final professional diagnostic impression using EXACTLY this format:

# ### 🩺 FINAL CLINICAL IMPRESSION

# **Primary Diagnosis:** [Most likely diagnosis]
# **Confidence:** [XX%]
# **Evidence Basis:** [One sentence citing specific visual findings and/or retrieved cases]

# **Differential Diagnoses:**
# 1. [Diagnosis] — [brief rationale]
# 2. [Diagnosis] — [brief rationale]

# **Recommended Next Steps:**
# - [Actionable clinical step]
# - [Actionable clinical step]

# **Urgency:** [Routine | Priority | Urgent | Emergent]"""

#         try:
#             synthesis = self.gemini.generate_content(prompt_synthesis).text
#         except Exception as e:
#             synthesis = f"Synthesis error: {e}"

#         return {
#             "visual":      visual_findings,
#             "correlation": correlation,
#             "synthesis":   synthesis,
#             "error":       None,
#         }

#     # ──────────────────────────────────────────
#     # Follow-up chat (stateful context window)
#     # ──────────────────────────────────────────
#     def answer_followup(
#         self,
#         question:   str,
#         report:     dict,
#         chat_history: list[dict],
#     ) -> str:
#         """
#         Answers follow-up questions with full diagnostic context injected.
#         chat_history: [{"role": "user"|"model", "parts": [str]}, ...]
#         """
#         if not self.gemini:
#             return "⚠️ Gemini unavailable."

#         system_ctx = f"""You are a clinical decision-support assistant.
# The following diagnostic report has already been generated for this patient.

# RADIOLOGIST FINDINGS:
# {report.get('visual', 'N/A')}

# CLINICAL CORRELATION:
# {report.get('correlation', 'N/A')}

# FINAL IMPRESSION:
# {report.get('synthesis', 'N/A')}

# Answer the clinician's follow-up question based strictly on the above context.
# Be concise, precise, and cite the report where relevant."""

#         # Prepend system context to first user turn
#         messages = [{"role": "user", "parts": [system_ctx]},
#                     {"role": "model", "parts": ["Understood. I am ready to answer follow-up questions about this case."]}]
#         messages += chat_history
#         messages.append({"role": "user", "parts": [question]})

#         try:
#             chat   = self.gemini.start_chat(history=messages[:-1])
#             reply  = chat.send_message(question)
#             return reply.text
#         except Exception as e:
#             return f"Error generating follow-up response: {e}"


# # ─────────────────────────────────────────────
# # CLI test harness
# # ─────────────────────────────────────────────
# if __name__ == "__main__":
#     copilot    = DiagnosticCopilot()
#     test_text  = "Patient presents with chronic cough and shortness of breath."
#     test_image = "data/images/CXR162_IM-0401-1001.png"

#     print("\n─── Retrieval ───")
#     cases = copilot.retrieve_similar_cases(text_query=test_text, image_path=test_image if os.path.exists(test_image) else None)
#     for c in cases:
#         print(f"  [{c['match_type']}] {c['xml_file']} → {c['score']:.4f}")

#     print("\n─── Diagnosis ───")
#     report = copilot.generate_diagnosis(test_text, test_image if os.path.exists(test_image) else None, cases)
#     if report.get("error"):
#         print(report["error"])
#     else:
#         print(report["synthesis"])

# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# from dotenv import load_dotenv
# load_dotenv()

# import torch
# import clip
# import numpy as np
# import cv2
# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from cortex import CortexClient
# import google.generativeai as genai
# from pytorch_grad_cam import GradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # ─────────────────────────────────────────────
# # Configuration
# # ─────────────────────────────────────────────
# DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# TEXT_MODEL    = "all-MiniLM-L6-v2"
# CLIP_MODEL    = "ViT-B/32"
# ACTIAN_ADDR   = "localhost:50051"

# # Gemini model priority list (most capable first)
# GEMINI_MODELS = [
#     "gemini-2.5-flash",
#     "gemini-2.0-flash-exp",
#     "gemini-1.5-flash-latest",
#     "gemini-1.5-flash",
#     "gemini-1.5-pro",
# ]


# class DiagnosticCopilot:
#     """
#     Multimodal RAG pipeline for chest X-ray clinical decision support.
    
#     Architecture:
#         1. CLIP ViT-B/32   → visual embeddings + GradCAM/EigenCAM heatmaps
#         2. all-MiniLM-L6   → text embeddings
#         3. Actian VectorAI → hybrid vector search (cosine)
#         4. Gemini (multi-agent) → visual analysis → clinical correlation → synthesis
#     """

#     def __init__(self):
#         print(f"[Init] Loading embedding models on {DEVICE}…")
#         self.text_model  = SentenceTransformer(TEXT_MODEL, device=DEVICE)
#         self.clip_model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
#         self.clip_model.eval()
#         self.gemini      = self._init_gemini()
#         print("[Init] DiagnosticCopilot ready.")

#     # ──────────────────────────────────────────
#     # Gemini initialisation (fail-fast, no I/O per attempt)
#     # ──────────────────────────────────────────
#     def _init_gemini(self):
#         api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             print("[Gemini] ❌  GEMINI_API_KEY not found.")
#             return None

#         genai.configure(api_key=api_key)

#         for name in GEMINI_MODELS:
#             try:
#                 model = genai.GenerativeModel(name)
#                 # Lightweight probe — avoids wasting quota on repeated "ping" calls
#                 model.generate_content(
#                     "Hi",
#                     generation_config=genai.GenerationConfig(max_output_tokens=1),
#                 )
#                 print(f"[Gemini] ✅  Connected → {name}")
#                 return model
#             except Exception as err:
#                 print(f"[Gemini] ⚠️  {name} unavailable: {err}")

#         print("[Gemini] ❌  All model variants failed.")
#         return None

#     # ──────────────────────────────────────────
#     # Explainable AI: Attention Heatmap
#     # ──────────────────────────────────────────
#     def generate_heatmap(self, image_path: str, text_query: str | None = None) -> str | None:
#         """
#         Generates an attention heatmap over the X-ray using CLIP's ViT-B/32.

#         Mode A — text_query provided: Text-Guided Grad-CAM
#             Computes gradients of cosine similarity between the IMAGE embedding
#             (post-projection, 512-d) and the TEXT embedding (post-projection, 512-d).
#             Both live in the same CLIP embedding space, so the similarity is valid.

#         Mode B — no text_query: EigenCAM (structural saliency, no gradient required)

#         Key fixes vs. the broken version:
#         ─────────────────────────────────
#         1. SemanticTarget operates on the PROJECTED image embedding (512-d) not the
#            raw CLS token (768-d), so the matmul against text_emb is dimensionally valid.
#            We do this by wrapping clip_model.visual WITH its projection layer using a
#            thin nn.Module wrapper so GradCAM still targets the last resblock internally.

#         2. reshape_transform: ViT-B/32 block output is [Tokens, B, C].
#            For 224×224 input with patch_size=32: grid = 224//32 = 7, tokens = 7×7+1 = 50.
#            Correct slice: t[1:] → [49, B, C], reshape to [7, 7, B, C], permute to [B, C, 7, 7].

#         3. Gradient flow: input tensor must retain grad. We call .requires_grad_(True)
#            before passing to GradCAM.

#         4. aug_smooth=True on GradCAM averages over slight augmentations, producing
#            significantly cleaner and more spatially focused heatmaps.
#         """
#         try:
#             raw    = Image.open(image_path).convert("RGB")
#             img_np = np.array(raw.resize((224, 224)), dtype=np.float32) / 255.0
#             tensor = self.preprocess(raw).unsqueeze(0).to(DEVICE)
#             tensor.requires_grad_(True)

#             # ── Correct reshape for ViT-B/32 ─────────────────────────────────
#             # Block output: [seq_len, batch, channels] = [50, 1, 768]
#             # Patch grid  : (224 / 32) = 7  →  7×7 = 49 patch tokens
#             # CLS token   : index 0  →  skip with [1:]
#             def reshape_transform(t):
#                 # t: [50, B, 768]
#                 patch_tokens = t[1:, :, :]          # [49, B, 768]
#                 B, C = patch_tokens.size(1), patch_tokens.size(2)
#                 # [49, B, C] → [B, C, 7, 7]
#                 patch_tokens = patch_tokens.permute(1, 2, 0)  # [B, C, 49]
#                 patch_tokens = patch_tokens.reshape(B, C, 7, 7)
#                 return patch_tokens

#             target_layers = [self.clip_model.visual.transformer.resblocks[-1]]

#             if text_query and text_query.strip():
#                 # ── Mode A: Text-Guided Grad-CAM ─────────────────────────────
#                 # Encode text into the shared 512-d CLIP space (with projection)
#                 tokens = clip.tokenize([text_query]).to(DEVICE)
#                 with torch.no_grad():
#                     t_emb = self.clip_model.encode_text(tokens).float()  # [1, 512]
#                     t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)

#                 # Wrapper: runs only the visual encoder's transformer blocks + projection
#                 # GradCAM hooks into resblocks[-1]; the wrapper's forward() must return
#                 # the projected 512-d embedding so the target function gets valid dims.
#                 import torch.nn as nn

#                 class _CLIPVisualWithProj(nn.Module):
#                     """Thin wrapper so GradCAM sees the projected embedding as output."""
#                     def __init__(self, visual_encoder):
#                         super().__init__()
#                         self.encoder = visual_encoder

#                     def forward(self, x):
#                         # Replicates clip.model.VisionTransformer.forward()
#                         x = self.encoder.conv1(x)                           # patch embed
#                         x = x.reshape(x.shape[0], x.shape[1], -1)
#                         x = x.permute(0, 2, 1)
#                         x = torch.cat([
#                             self.encoder.class_embedding.to(x.dtype) +
#                             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#                             x
#                         ], dim=1)
#                         x = x + self.encoder.positional_embedding.to(x.dtype)
#                         x = self.encoder.ln_pre(x)
#                         x = x.permute(1, 0, 2)          # [seq, B, C]
#                         x = self.encoder.transformer(x) # runs all resblocks (hooks fire here)
#                         x = x.permute(1, 0, 2)          # [B, seq, C]
#                         x = self.encoder.ln_post(x[:, 0, :])   # CLS token + LayerNorm
#                         if self.encoder.proj is not None:
#                             x = x @ self.encoder.proj   # project to 512-d  ← key fix
#                         return x                         # [B, 512]

#                 wrapped = _CLIPVisualWithProj(self.clip_model.visual).to(DEVICE)
#                 wrapped.eval()

#                 # Target: maximise cosine similarity with text embedding in 512-d space
#                 class _SemanticTarget:
#                     def __init__(self, emb): self.emb = emb          # [1, 512]
#                     def __call__(self, model_output):                 # model_output: [B, 512]
#                         img_norm = model_output / (model_output.norm(dim=-1, keepdim=True) + 1e-8)
#                         return img_norm @ self.emb.t()                # [B, 1] → scalar per batch

#                 cam = GradCAM(
#                     model=wrapped,
#                     target_layers=[wrapped.encoder.transformer.resblocks[-1]],
#                     reshape_transform=reshape_transform,
#                 )
#                 grayscale = cam(
#                     input_tensor=tensor,
#                     targets=[_SemanticTarget(t_emb)],
#                     aug_smooth=True,          # averages over augmentations → cleaner map
#                     eigen_smooth=True,        # PCA denoising → removes background noise
#                 )[0]
#                 print(f"[Heatmap] Grad-CAM for: '{text_query}'  "
#                       f"range=[{grayscale.min():.3f}, {grayscale.max():.3f}]")

#             else:
#                 # ── Mode B: EigenCAM — no gradient needed, always works ───────
#                 cam = EigenCAM(
#                     model=self.clip_model.visual,
#                     target_layers=target_layers,
#                     reshape_transform=reshape_transform,
#                 )
#                 grayscale = cam(input_tensor=tensor)[0]
#                 print(f"[Heatmap] EigenCAM  "
#                       f"range=[{grayscale.min():.3f}, {grayscale.max():.3f}]")

#             # ── Normalise with histogram stretch for maximum contrast ─────────
#             lo, hi = np.percentile(grayscale, 2), np.percentile(grayscale, 98)
#             grayscale = np.clip((grayscale - lo) / (hi - lo + 1e-8), 0, 1)

#             # ── Overlay on original image ─────────────────────────────────────
#             vis = show_cam_on_image(img_np, grayscale, use_rgb=True, colormap=cv2.COLORMAP_INFERNO)

#             out_path = os.path.join(os.getcwd(), f"heatmap_{os.path.basename(image_path)}")
#             cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
#             print(f"[Heatmap] Saved → {out_path}")
#             return out_path

#         except Exception as e:
#             print(f"[Heatmap] Error: {e}")
#             import traceback; traceback.print_exc()
#             return None

#     # ──────────────────────────────────────────
#     # Hybrid Retrieval (Text + Visual)
#     # ──────────────────────────────────────────
#     def retrieve_similar_cases(
#         self,
#         text_query: str | None = None,
#         image_path: str | None = None,
#         top_k: int = 3,
#     ) -> list[dict]:
#         """
#         Hybrid retrieval from Actian VectorAI:
#           - Semantic text search  → cxr_text   (MiniLM, dim=384)
#           - Visual image search   → cxr_images (CLIP,   dim=512)
#         Results are score-sorted and deduplicated by xml_file.
#         """
#         results = []

#         with CortexClient(ACTIAN_ADDR) as client:

#             # 1. Text branch
#             if text_query and text_query.strip():
#                 vec = self.text_model.encode([text_query])[0].tolist()
#                 for hit in client.search("cxr_text", vec, top_k=top_k):
#                     _, payload = client.get("cxr_text", hit.id)
#                     results.append({
#                         "match_type": "📝 Semantic",
#                         "score":      hit.score,
#                         "xml_file":   payload.get("xml_file", "unknown"),
#                         "impression": payload.get("impression", ""),
#                         "path":       None,
#                     })

#             # 2. Visual branch
#             if image_path and os.path.exists(image_path):
#                 img = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
#                 with torch.no_grad():
#                     feats = self.clip_model.encode_image(img)
#                     feats = feats / feats.norm(dim=-1, keepdim=True)
#                     vec   = feats.cpu().numpy().flatten().tolist()

#                 for hit in client.search("cxr_images", vec, top_k=top_k):
#                     _, payload = client.get("cxr_images", hit.id)
#                     results.append({
#                         "match_type": "🖼️  Visual",
#                         "score":      hit.score,
#                         "xml_file":   payload.get("xml_file", "unknown"),
#                         "impression": payload.get("impression", ""),
#                         "path":       payload.get("path"),
#                     })

#         # Sort by score, deduplicate by xml_file (keep highest score)
#         results.sort(key=lambda x: x["score"], reverse=True)
#         seen, deduped = set(), []
#         for r in results:
#             if r["xml_file"] not in seen:
#                 seen.add(r["xml_file"])
#                 deduped.append(r)

#         return deduped[:top_k]

#     # ──────────────────────────────────────────
#     # Multi-Agent Diagnostic Pipeline
#     # ──────────────────────────────────────────
#     def generate_diagnosis(
#         self,
#         clinical_notes: str,
#         image_path:     str | None,
#         retrieved_cases: list[dict],
#     ) -> dict:
#         """
#         Three-agent pipeline:
#           Agent 1 → Visual Radiologist   (image-grounded)
#           Agent 2 → Clinical Integrator  (notes + evidence)
#           Agent 3 → Chief Synthesis      (final impression)

#         Returns a dict with keys: visual, correlation, synthesis, error
#         """
#         if not self.gemini:
#             self.gemini = self._init_gemini()
#         if not self.gemini:
#             return {
#                 "error": (
#                     "⚠️ **Gemini Unavailable** — Set `GEMINI_API_KEY` in your `.env` file "
#                     "and restart the app."
#                 )
#             }

#         # Build evidence block
#         evidence_block = "\n".join(
#             f"  [{i+1}] {c['xml_file']} (score {c['score']:.4f}): "
#             f"{c.get('impression') or 'visual pattern match only'}"
#             for i, c in enumerate(retrieved_cases)
#         ) or "  No cases retrieved."

#         # ── Agent 1: Visual Radiologist ───────────────────────────────────────
#         visual_findings = "No radiograph provided."
#         if image_path and os.path.exists(image_path):
#             try:
#                 img = Image.open(image_path)
#                 prompt_visual = (
#                     "You are an expert chest radiologist. "
#                     "Provide a structured, objective technical analysis of this X-ray. "
#                     "Cover: cardiac silhouette, pulmonary vascularity, lung parenchyma, "
#                     "pleural spaces, bony structures, mediastinum, and any incidental findings. "
#                     "Be precise and avoid speculation."
#                 )
#                 visual_findings = self.gemini.generate_content([prompt_visual, img]).text
#             except Exception as e:
#                 visual_findings = f"Visual analysis error: {e}"

#         # ── Agent 2: Clinical Integrator ─────────────────────────────────────
#         prompt_integrate = f"""You are a Clinical Integration Specialist.

# VISUAL FINDINGS (from Radiologist Agent):
# {visual_findings}

# PATIENT CLINICAL NOTES:
# {clinical_notes or "None provided."}

# ACTIAN VECTORAI EVIDENCE BASE (top similar historical cases):
# {evidence_block}

# TASK:
# Cross-reference the visual findings and clinical notes against the retrieved historical cases.
# Identify convergent diagnostic patterns. Note any discrepancies or atypical features.
# Generate a ranked differential diagnosis with supporting evidence citations (e.g., "[Case 2]").
# Keep the tone clinical and concise."""

#         try:
#             correlation = self.gemini.generate_content(prompt_integrate).text
#         except Exception as e:
#             correlation = f"Clinical correlation error: {e}"

#         # ── Agent 3: Chief of Medicine Synthesis ──────────────────────────────
#         prompt_synthesis = f"""You are the Chief of Medicine conducting final case review.

# CLINICAL CORRELATION ANALYSIS:
# {correlation}

# Generate a final professional diagnostic impression using EXACTLY this format:

# ### 🩺 FINAL CLINICAL IMPRESSION

# **Primary Diagnosis:** [Most likely diagnosis]
# **Confidence:** [XX%]
# **Evidence Basis:** [One sentence citing specific visual findings and/or retrieved cases]

# **Differential Diagnoses:**
# 1. [Diagnosis] — [brief rationale]
# 2. [Diagnosis] — [brief rationale]

# **Recommended Next Steps:**
# - [Actionable clinical step]
# - [Actionable clinical step]

# **Urgency:** [Routine | Priority | Urgent | Emergent]"""

#         try:
#             synthesis = self.gemini.generate_content(prompt_synthesis).text
#         except Exception as e:
#             synthesis = f"Synthesis error: {e}"

#         return {
#             "visual":      visual_findings,
#             "correlation": correlation,
#             "synthesis":   synthesis,
#             "error":       None,
#         }

#     # ──────────────────────────────────────────
#     # Follow-up chat (stateful context window)
#     # ──────────────────────────────────────────
#     def answer_followup(
#         self,
#         question:   str,
#         report:     dict,
#         chat_history: list[dict],
#     ) -> str:
#         """
#         Answers follow-up questions with full diagnostic context injected.
#         chat_history: [{"role": "user"|"model", "parts": [str]}, ...]
#         """
#         if not self.gemini:
#             return "⚠️ Gemini unavailable."

#         system_ctx = f"""You are a clinical decision-support assistant.
# The following diagnostic report has already been generated for this patient.

# RADIOLOGIST FINDINGS:
# {report.get('visual', 'N/A')}

# CLINICAL CORRELATION:
# {report.get('correlation', 'N/A')}

# FINAL IMPRESSION:
# {report.get('synthesis', 'N/A')}

# Answer the clinician's follow-up question based strictly on the above context.
# Be concise, precise, and cite the report where relevant."""

#         # Prepend system context to first user turn
#         messages = [{"role": "user", "parts": [system_ctx]},
#                     {"role": "model", "parts": ["Understood. I am ready to answer follow-up questions about this case."]}]
#         messages += chat_history
#         messages.append({"role": "user", "parts": [question]})

#         try:
#             chat   = self.gemini.start_chat(history=messages[:-1])
#             reply  = chat.send_message(question)
#             return reply.text
#         except Exception as e:
#             return f"Error generating follow-up response: {e}"


# # ─────────────────────────────────────────────
# # CLI test harness
# # ─────────────────────────────────────────────
# if __name__ == "__main__":
#     copilot    = DiagnosticCopilot()
#     test_text  = "Patient presents with chronic cough and shortness of breath."
#     test_image = "data/images/CXR162_IM-0401-1001.png"

#     print("\n─── Retrieval ───")
#     cases = copilot.retrieve_similar_cases(text_query=test_text, image_path=test_image if os.path.exists(test_image) else None)
#     for c in cases:
#         print(f"  [{c['match_type']}] {c['xml_file']} → {c['score']:.4f}")

#     print("\n─── Diagnosis ───")
#     report = copilot.generate_diagnosis(test_text, test_image if os.path.exists(test_image) else None, cases)
#     if report.get("error"):
#         print(report["error"])
#     else:
#         print(report["synthesis"])

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import clip
import numpy as np
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
from cortex import CortexClient
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
#
#   Set LLM_PROVIDER to switch backends:
#     "gemini" — Google Gemini API  (requires GEMINI_API_KEY in .env)
#     "ollama" — Local Ollama models (free, no API key, runs on your Mac)
#
#   When you get your Gemini credits back: set LLM_PROVIDER = "gemini"
#   OR set the env var:  export LLM_PROVIDER=gemini
#
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "gemini" | "ollama"

# Ollama model config
OLLAMA_VISION_MODEL = "llava:13b"   # swap to llava:7b if you have <16GB RAM
OLLAMA_TEXT_MODEL   = "mistral"     # text-only agents (faster, lower memory)

# Gemini model priority list (most capable → fallback)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Embedding + retrieval config
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL  = "ViT-B/32"
ACTIAN_ADDR = "localhost:50051"


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROVIDER ABSTRACTION
# Two classes with an identical interface: vision_text(), text(), chat()
# DiagnosticCopilot only ever calls these three methods — never touches
# provider-specific APIs directly.
# ─────────────────────────────────────────────────────────────────────────────

class _GeminiProvider:
    """Google Gemini API backend."""

    def __init__(self):
        import google.generativeai as genai
        self._genai = genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Add it to your .env file:\n"
                "  GEMINI_API_KEY=your_key_here"
            )

        genai.configure(api_key=api_key)
        self.model = self._connect()

    def _connect(self):
        for name in GEMINI_MODELS:
            try:
                m = self._genai.GenerativeModel(name)
                m.generate_content(
                    "ping",
                    generation_config=self._genai.GenerationConfig(max_output_tokens=1),
                )
                print(f"[Gemini] ✅  Connected → {name}")
                return m
            except Exception as e:
                print(f"[Gemini] ⚠️  {name} failed: {e}")
        raise RuntimeError(
            "All Gemini model variants failed.\n"
            "  → Check your API key and quota at: https://aistudio.google.com"
        )

    def vision_text(self, prompt: str, image: Image.Image) -> str:
        """Image + text → response string."""
        return self.model.generate_content([prompt, image]).text

    def text(self, prompt: str) -> str:
        """Text-only → response string."""
        return self.model.generate_content(prompt).text

    def chat(self, messages: list[dict]) -> str:
        """
        Multi-turn chat.
        messages: [{"role": "user"|"model", "parts": [str]}, ...]
        """
        history = messages[:-1]
        last    = messages[-1]["parts"][0]
        session = self.model.start_chat(history=history)
        return session.send_message(last).text


class _OllamaProvider:
    """Local Ollama backend. Compatible interface with _GeminiProvider."""

    def __init__(self):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise RuntimeError(
                "Ollama Python client not installed.\n"
                "  Run: pip install ollama"
            )
        self._verify_connection()

    def _verify_connection(self):
        try:
            self._ollama.chat(
                model=OLLAMA_TEXT_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                options={"num_predict": 1},
            )
            print(
                f"[Ollama] ✅  Connected — "
                f"vision={OLLAMA_VISION_MODEL}, text={OLLAMA_TEXT_MODEL}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama not reachable: {e}\n"
                f"  → Start Ollama:     ollama serve\n"
                f"  → Pull models:      ollama pull {OLLAMA_VISION_MODEL}\n"
                f"                      ollama pull {OLLAMA_TEXT_MODEL}"
            )

    def vision_text(self, prompt: str, image: Image.Image) -> str:
        """
        Ollama needs an image file path. We write a temp file from the
        PIL Image, call Ollama, then clean up.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        try:
            r = self._ollama.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[{
                    "role":    "user",
                    "content": prompt,
                    "images":  [tmp_path],
                }],
            )
            return r["message"]["content"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def text(self, prompt: str) -> str:
        """Text-only call using the faster text model."""
        r = self._ollama.chat(
            model=OLLAMA_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return r["message"]["content"]

    def chat(self, messages: list[dict]) -> str:
        """
        Multi-turn chat. Converts from Gemini-style parts format
        {"role": "model", "parts": [str]} to Ollama format
        {"role": "assistant", "content": str} internally.
        """
        ollama_messages = []
        for m in messages:
            role    = "assistant" if m["role"] == "model" else m["role"]
            content = (
                m["parts"][0]
                if isinstance(m.get("parts"), list)
                else m.get("content", "")
            )
            ollama_messages.append({"role": role, "content": content})

        r = self._ollama.chat(
            model=OLLAMA_TEXT_MODEL,
            messages=ollama_messages,
        )
        return r["message"]["content"]


def _build_provider() -> "_GeminiProvider | _OllamaProvider":
    """Factory — instantiates the provider set in LLM_PROVIDER."""
    provider = LLM_PROVIDER.strip().lower()
    if provider == "gemini":
        return _GeminiProvider()
    elif provider == "ollama":
        return _OllamaProvider()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. "
            "Valid values: 'gemini' or 'ollama'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLIP VISUAL WRAPPER FOR GRADCAM
# ─────────────────────────────────────────────────────────────────────────────

class _CLIPVisualWithProj(nn.Module):
    """
    Thin wrapper around CLIP's visual encoder that includes the final
    projection layer in its forward pass.

    WHY THIS EXISTS:
    GradCAM hooks into resblocks[-1] and receives its output as the
    "model_output" passed to the target function. Without this wrapper,
    clip_model.visual returns the raw CLS token at 768-d. The text
    embedding from encode_text() is 512-d (post-projection). Doing a
    matmul between 768-d and 512-d is either an error or silent garbage,
    producing zero/uniform gradients → flat heatmap.

    This wrapper replicates VisionTransformer.forward() exactly, ending
    with `x @ self.proj` so the output is 512-d and lives in the same
    embedding space as the text embedding.
    """

    def __init__(self, visual_encoder):
        super().__init__()
        self.encoder = visual_encoder

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.encoder.class_embedding.to(x.dtype)
            + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x,
        ], dim=1)
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)            # [seq, B, C]
        x = self.encoder.transformer(x)   # GradCAM hooks fire here
        x = x.permute(1, 0, 2)            # [B, seq, C]
        x = self.encoder.ln_post(x[:, 0, :])  # CLS token + LayerNorm
        if self.encoder.proj is not None:
            x = x @ self.encoder.proj     # → [B, 512]  ← the critical fix
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticCopilot:
    """
    Multimodal RAG pipeline for chest X-ray clinical decision support.

    Architecture:
        1. CLIP ViT-B/32     → visual embeddings + GradCAM/EigenCAM heatmaps
        2. all-MiniLM-L6-v2  → text embeddings for semantic search
        3. Actian VectorAI   → hybrid cosine vector search (text + image)
        4. LLM backend       → 3-agent diagnostic pipeline
                               (Gemini or Ollama, hot-swappable)
    """

    def __init__(self):
        print(f"[Init] Device={DEVICE} | Provider={LLM_PROVIDER.upper()}")

        # Embedding models (always loaded, provider-agnostic)
        self.text_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.clip_model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
        self.clip_model.eval()

        # Pre-build GradCAM-compatible CLIP wrapper (reused across calls)
        self._clip_proj = _CLIPVisualWithProj(self.clip_model.visual).to(DEVICE)
        self._clip_proj.eval()

        # LLM provider
        self.llm = self._init_llm()
        print("[Init] DiagnosticCopilot ready.")

    def _init_llm(self):
        try:
            return _build_provider()
        except Exception as e:
            print(f"[LLM] ⚠️  Provider init failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Hot-swap provider at runtime (e.g. from Streamlit sidebar toggle)
    # ─────────────────────────────────────────────────────────────────────────

    def switch_provider(self, provider: str) -> bool:
        """
        Switch LLM backend without restarting the app.
        Returns True if the new provider connected successfully.

        Usage:
            copilot.switch_provider("gemini")   # when credits are back
            copilot.switch_provider("ollama")   # back to local
        """
        global LLM_PROVIDER
        LLM_PROVIDER = provider
        print(f"[LLM] Switching provider → {provider.upper()}")
        self.llm = self._init_llm()
        return self.llm is not None

    @property
    def active_provider(self) -> str:
        return LLM_PROVIDER.upper()

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP — Explainable AI
    # ─────────────────────────────────────────────────────────────────────────

    def generate_heatmap(
        self,
        image_path:  str,
        text_query:  str | None = None,
    ) -> str | None:
        """
        Generates a clinical attention heatmap over the X-ray.

        Mode A — text_query provided: Text-Guided Grad-CAM
            Gradient of cosine similarity between image embedding (512-d,
            post-projection) and text embedding (512-d). Both in the same
            CLIP space — dot product is meaningful.

        Mode B — no text_query: EigenCAM
            PCA-based structural saliency. No gradients required.
            Always produces a valid map, good fallback.

        Fixed issues vs. original code:
            1. Projection mismatch (768 vs 512): fixed via _CLIPVisualWithProj
            2. reshape_transform: [50,B,C] → skip CLS → [B,C,7,7] (correct)
            3. Percentile normalisation instead of min-max (better contrast)
            4. aug_smooth + eigen_smooth for cleaner clinical maps
        """
        try:
            raw    = Image.open(image_path).convert("RGB")
            img_np = np.array(raw.resize((224, 224)), dtype=np.float32) / 255.0
            tensor = self.preprocess(raw).unsqueeze(0).to(DEVICE)
            tensor.requires_grad_(True)

            # ViT-B/32: patch_size=32, input=224x224
            # → grid = 224/32 = 7x7 = 49 patch tokens + 1 CLS = 50 total
            # Block output shape: [50, B, 768]
            def reshape_transform(t):
                patches = t[1:, :, :]                      # [49, B, 768] — drop CLS
                B, C    = patches.size(1), patches.size(2)
                patches = patches.permute(1, 2, 0)         # [B, C, 49]
                return patches.reshape(B, C, 7, 7)         # [B, C, 7, 7]

            if text_query and text_query.strip():
                # ── Mode A: Text-Guided Grad-CAM ─────────────────────────────
                tokens = clip.tokenize([text_query]).to(DEVICE)
                with torch.no_grad():
                    t_emb = self.clip_model.encode_text(tokens).float()  # [1, 512]
                    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)

                class _SemanticTarget:
                    """Maximise cosine similarity between image and text embeddings."""
                    def __init__(self, emb): self.emb = emb   # [1, 512]
                    def __call__(self, out):                   # out: [B, 512]
                        img_n = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
                        return img_n @ self.emb.t()            # [B, 1]

                cam = GradCAM(
                    model=self._clip_proj,
                    target_layers=[self._clip_proj.encoder.transformer.resblocks[-1]],
                    reshape_transform=reshape_transform,
                )
                grayscale = cam(
                    input_tensor=tensor,
                    targets=[_SemanticTarget(t_emb)],
                    aug_smooth=True,    # average over augmentations → smoother
                    eigen_smooth=True,  # PCA denoising → less background noise
                )[0]
                print(
                    f"[Heatmap] Grad-CAM | query='{text_query}' | "
                    f"range=[{grayscale.min():.3f}, {grayscale.max():.3f}]"
                )

            else:
                # ── Mode B: EigenCAM ─────────────────────────────────────────
                cam = EigenCAM(
                    model=self.clip_model.visual,
                    target_layers=[self.clip_model.visual.transformer.resblocks[-1]],
                    reshape_transform=reshape_transform,
                )
                grayscale = cam(input_tensor=tensor)[0]
                print(
                    f"[Heatmap] EigenCAM | "
                    f"range=[{grayscale.min():.3f}, {grayscale.max():.3f}]"
                )

            # Percentile normalisation — robust to outliers, maximises contrast
            lo = np.percentile(grayscale, 2)
            hi = np.percentile(grayscale, 98)
            grayscale = np.clip((grayscale - lo) / (hi - lo + 1e-8), 0, 1)

            vis = show_cam_on_image(
                img_np,
                grayscale,
                use_rgb=True,
                colormap=cv2.COLORMAP_INFERNO,  # better perceptual contrast than JET
            )

            out_path = os.path.join(
                os.getcwd(), f"heatmap_{os.path.basename(image_path)}"
            )
            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"[Heatmap] Saved → {out_path}")
            return out_path

        except Exception as e:
            print(f"[Heatmap] Error: {e}")
            import traceback; traceback.print_exc()
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVAL — Actian VectorAI hybrid search
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_similar_cases(
        self,
        text_query:  str | None = None,
        image_path:  str | None = None,
        top_k:       int = 3,
    ) -> list[dict]:
        """
        Hybrid retrieval from Actian VectorAI.

        Text branch  → cxr_text   collection (MiniLM 384-d, cosine)
        Visual branch → cxr_images collection (CLIP   512-d, cosine)

        Results are merged, score-sorted, and deduplicated by xml_file
        (keeping the highest score per case when both branches hit the same file).
        """
        results = []

        with CortexClient(ACTIAN_ADDR) as client:

            # Text branch
            if text_query and text_query.strip():
                vec = self.text_model.encode([text_query])[0].tolist()
                for hit in client.search("cxr_text", vec, top_k=top_k):
                    _, payload = client.get("cxr_text", hit.id)
                    results.append({
                        "match_type": "📝 Semantic",
                        "score":      hit.score,
                        "xml_file":   payload.get("xml_file", "unknown"),
                        "impression": payload.get("impression", ""),
                        "path":       None,
                    })

            # Visual branch
            if image_path and os.path.exists(image_path):
                img = self.preprocess(
                    Image.open(image_path)
                ).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feats = self.clip_model.encode_image(img)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    vec   = feats.cpu().numpy().flatten().tolist()

                for hit in client.search("cxr_images", vec, top_k=top_k):
                    _, payload = client.get("cxr_images", hit.id)
                    results.append({
                        "match_type": "🖼️  Visual",
                        "score":      hit.score,
                        "xml_file":   payload.get("xml_file", "unknown"),
                        "impression": payload.get("impression", ""),
                        "path":       payload.get("path"),
                    })

        # Sort → deduplicate (highest score wins per xml_file)
        results.sort(key=lambda x: x["score"], reverse=True)
        seen, deduped = set(), []
        for r in results:
            if r["xml_file"] not in seen:
                seen.add(r["xml_file"])
                deduped.append(r)

        return deduped[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSIS — 3-Agent Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def generate_diagnosis(
        self,
        clinical_notes:  str,
        image_path:      str | None,
        retrieved_cases: list[dict],
    ) -> dict:
        """
        Three-agent diagnostic pipeline — fully provider-agnostic.

          Agent 1 → Visual Radiologist  (image + text → llm.vision_text)
          Agent 2 → Clinical Integrator (text only    → llm.text)
          Agent 3 → Chief Synthesis     (text only    → llm.text)

        Returns dict:
            visual      — Agent 1 output
            correlation — Agent 2 output
            synthesis   — Agent 3 output
            provider    — which LLM was used
            error       — None on success, error string on failure
        """
        # Lazy re-init if startup failed (e.g. Ollama wasn't running yet)
        if not self.llm:
            self.llm = self._init_llm()
        if not self.llm:
            tip = (
                "Set `GEMINI_API_KEY` in your `.env` file."
                if LLM_PROVIDER == "gemini"
                else "Run `ollama serve` and ensure models are pulled:\n"
                     f"  ollama pull {OLLAMA_VISION_MODEL}\n"
                     f"  ollama pull {OLLAMA_TEXT_MODEL}"
            )
            return {"error": f"⚠️ **{LLM_PROVIDER.upper()} unavailable.**\n\n{tip}"}

        # Build evidence context block
        evidence_block = "\n".join(
            f"  [{i+1}] {c['xml_file']} (score {c['score']:.4f}): "
            f"{c.get('impression') or 'visual pattern match only'}"
            for i, c in enumerate(retrieved_cases)
        ) or "  No cases retrieved."

        # ── Agent 1: Visual Radiologist ───────────────────────────────────────
        visual_findings = "No radiograph provided."
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                prompt_visual = (
                    "You are an expert chest radiologist. "
                    "Provide a structured, objective technical analysis of this X-ray. "
                    "Cover: cardiac silhouette, pulmonary vascularity, lung parenchyma, "
                    "pleural spaces, bony structures, mediastinum, and any incidental findings. "
                    "Be precise and avoid speculation."
                )
                visual_findings = self.llm.vision_text(prompt_visual, img)
            except Exception as e:
                visual_findings = f"Visual analysis error: {e}"

        # ── Agent 2: Clinical Integrator ─────────────────────────────────────
        prompt_integrate = f"""You are a Clinical Integration Specialist.

VISUAL FINDINGS (Radiologist Agent):
{visual_findings}

PATIENT CLINICAL NOTES:
{clinical_notes or "None provided."}

ACTIAN VECTORAI EVIDENCE BASE (retrieved similar historical cases):
{evidence_block}

TASK:
Cross-reference the visual findings and clinical notes against the retrieved historical cases.
Identify convergent diagnostic patterns and note any discrepancies.
Generate a ranked differential diagnosis with evidence citations e.g. "[Case 2]".
Keep the tone clinical and concise."""

        try:
            correlation = self.llm.text(prompt_integrate)
        except Exception as e:
            correlation = f"Clinical correlation error: {e}"

        # ── Agent 3: Chief of Medicine Synthesis ──────────────────────────────
        prompt_synthesis = f"""You are the Chief of Medicine conducting final case review.

CLINICAL CORRELATION ANALYSIS:
{correlation}

Generate a final professional diagnostic impression using EXACTLY this format:

### 🩺 FINAL CLINICAL IMPRESSION

**Primary Diagnosis:** [Most likely diagnosis]
**Confidence:** [XX%]
**Evidence Basis:** [One sentence citing specific visual findings and/or retrieved cases]

**Differential Diagnoses:**
1. [Diagnosis] — [brief rationale]
2. [Diagnosis] — [brief rationale]

**Recommended Next Steps:**
- [Actionable clinical step]
- [Actionable clinical step]

**Urgency:** [Routine | Priority | Urgent | Emergent]"""

        try:
            synthesis = self.llm.text(prompt_synthesis)
        except Exception as e:
            synthesis = f"Synthesis error: {e}"

        return {
            "visual":      visual_findings,
            "correlation": correlation,
            "synthesis":   synthesis,
            "provider":    LLM_PROVIDER,
            "error":       None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # FOLLOW-UP CHAT
    # ─────────────────────────────────────────────────────────────────────────

    def answer_followup(
        self,
        question:     str,
        report:       dict,
        chat_history: list[dict],
    ) -> str:
        """
        Stateful follow-up Q&A with the full diagnostic report as context.

        chat_history: [{"role": "user"|"model", "parts": [str]}, ...]
        _OllamaProvider.chat() handles the format conversion internally.
        """
        if not self.llm:
            return "⚠️ LLM provider unavailable. Check your configuration."

        system_ctx = f"""You are a clinical decision-support assistant.
A diagnostic report has already been generated for this patient.
Answer follow-up questions based strictly on the report. Be concise and cite it.

RADIOLOGIST FINDINGS:
{report.get('visual', 'N/A')}

CLINICAL CORRELATION:
{report.get('correlation', 'N/A')}

FINAL IMPRESSION:
{report.get('synthesis', 'N/A')}"""

        messages = [
            {"role": "user",  "parts": [system_ctx]},
            {"role": "model", "parts": ["Understood. Ready for follow-up questions."]},
        ]
        messages += chat_history
        messages.append({"role": "user", "parts": [question]})

        try:
            return self.llm.chat(messages)
        except Exception as e:
            return f"Error generating response: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST HARNESS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    copilot    = DiagnosticCopilot()
    test_text  = "Patient presents with chronic cough and shortness of breath."
    test_image = "data/images/CXR162_IM-0401-1001.png"

    print(f"\n─── Active Provider: {LLM_PROVIDER.upper()} ───")

    print("\n─── Retrieval ───")
    cases = copilot.retrieve_similar_cases(
        text_query=test_text,
        image_path=test_image if os.path.exists(test_image) else None,
    )
    for c in cases:
        print(f"  [{c['match_type']}] {c['xml_file']} → {c['score']:.4f}")

    print("\n─── Heatmap ───")
    if os.path.exists(test_image):
        hp = copilot.generate_heatmap(test_image, text_query=test_text)
        print(f"  Saved: {hp}")

    print("\n─── Diagnosis ───")
    report = copilot.generate_diagnosis(
        test_text,
        test_image if os.path.exists(test_image) else None,
        cases,
    )
    if report.get("error"):
        print(report["error"])
    else:
        print(report["synthesis"])