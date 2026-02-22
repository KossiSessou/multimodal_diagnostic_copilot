import pandas as pd
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from cortex import CortexClient, DistanceMetric
from tqdm import tqdm
import os
import argparse

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "ViT-B/32"
ACTIAN_ADDR = "localhost:50051"
BATCH_SIZE = 32

def index_modality(csv_path, collection_prefix):
    """
    Indexes data into modality-specific collections (e.g., cxr_text, dermo_text).
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path).head(1000) # Indexing 1000 for each for the demo
    print(f"--- 🔄 Indexing {len(df)} records for {collection_prefix} ---")

    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)

    with CortexClient(ACTIAN_ADDR) as client:
        # Create unique collections for this modality
        text_col = f"{collection_prefix}_text"
        img_col = f"{collection_prefix}_images"
        
        client.create_collection(name=text_col, dimension=384, distance_metric=DistanceMetric.COSINE)
        client.create_collection(name=img_col, dimension=512, distance_metric=DistanceMetric.COSINE)

        for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Indexing {collection_prefix}"):
            batch = df.iloc[i : i + BATCH_SIZE]
            
            # Text Indexing
            texts = batch["findings"].fillna("").tolist()
            text_embeddings = text_model.encode(texts).tolist()
            text_ids = [i + j for j in range(len(text_embeddings))]
            text_payloads = [{"xml_file": str(row["xml_file"]), "findings": str(row["findings"])} for _, row in batch.iterrows()]
            client.batch_upsert(collection_name=text_col, ids=text_ids, vectors=text_embeddings, payloads=text_payloads)

            # Image Indexing
            img_embeddings = []
            img_ids = []
            img_payloads = []
            for idx, row in batch.iterrows():
                try:
                    img = preprocess(Image.open(row["image_paths"])).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        feat = clip_model.encode_image(img)
                        feat /= feat.norm(dim=-1, keepdim=True)
                        img_embeddings.append(feat.cpu().numpy().flatten().tolist())
                        img_ids.append(1000000 + idx)
                        img_payloads.append({"xml_file": str(row["xml_file"]), "findings": str(row["findings"]), "path": str(row["image_paths"])})
                except: continue
            
            if img_embeddings:
                client.batch_upsert(collection_name=img_col, ids=img_ids, vectors=img_embeddings, payloads=img_payloads)

if __name__ == "__main__":
    # 1. Index X-Rays (if the processed_data.csv is the X-ray one)
    # index_modality("processed_data_cxr.csv", "cxr")
    
    # 2. Index Skin (if the processed_data.csv is the HAM10000 one)
    # index_modality("processed_data_ham.csv", "dermo")
    
    print("Run index_modality manually for each dataset to keep them separate.")
