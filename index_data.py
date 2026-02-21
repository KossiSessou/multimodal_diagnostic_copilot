import pandas as pd
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from cortex import CortexClient, DistanceMetric
from tqdm import tqdm
import os

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "ViT-B/32"
ACTIAN_ADDR = "localhost:50051"
BATCH_SIZE = 32

def main():
    # 1. Load the processed data
    if not os.path.exists("processed_data.csv"):
        print("Error: processed_data.csv not found. Run Phase 2 first.")
        return
    
    df = pd.read_csv("processed_data.csv").head(500) # Limiting to 500 for initial test
    print(f"Loaded {len(df)} records for indexing.")

    # 2. Initialize Embedding Models
    print(f"Loading models on {DEVICE}...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)

    # 3. Connect to Actian VectorAI DB
    print(f"Connecting to Actian VectorAI DB at {ACTIAN_ADDR}...")
    with CortexClient(ACTIAN_ADDR) as client:
        # Create Collections
        # Dimensions: all-MiniLM-L6-v2 = 384, CLIP ViT-B/32 = 512
        print("Creating collections...")
        try:
            client.create_collection(
                name="cxr_text", 
                dimension=384, 
                distance_metric=DistanceMetric.COSINE
            )
            client.create_collection(
                name="cxr_images", 
                dimension=512, 
                distance_metric=DistanceMetric.COSINE
            )
        except Exception as e:
            print(f"Note: Collections might already exist: {e}")

        # 4. Process and Index in Batches
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Indexing Batches"):
            batch = df.iloc[i : i + BATCH_SIZE]
            
            # --- Text Embeddings ---
            texts = batch["findings"].fillna("").tolist()
            text_embeddings = text_model.encode(texts).tolist()
            
            # --- Image Embeddings ---
            image_embeddings = []
            valid_indices = []
            
            for idx, img_path in enumerate(batch["image_paths"]):
                try:
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        image_embeddings.append(image_features.cpu().numpy().flatten().tolist())
                        valid_indices.append(idx)
                except Exception as e:
                    print(f"Skip image {img_path}: {e}")

            # --- Batch Upsert to Actian ---
            # Index Text
            text_ids = [i + j for j in range(len(text_embeddings))]
            text_payloads = [
                {
                    "xml_file": str(batch.iloc[j]["xml_file"]),
                    "impression": str(batch.iloc[j]["impression"])[:500] if pd.notna(batch.iloc[j]["impression"]) else ""
                }
                for j in range(len(text_embeddings))
            ]
            
            client.batch_upsert(
                collection_name="cxr_text",
                ids=text_ids,
                vectors=text_embeddings,
                payloads=text_payloads
            )

            # Index Images
            if image_embeddings:
                # Use a high offset for image IDs to keep them distinct from text IDs if needed
                # Or just use the dataframe index if it's unique
                img_ids = [1000000 + i + idx for idx in valid_indices]
                img_payloads = [
                    {
                        "xml_file": str(batch.loc[batch.index[idx], "xml_file"]),
                        "path": str(batch.loc[batch.index[idx], "image_paths"])
                    }
                    for idx in valid_indices
                ]
                
                client.batch_upsert(
                    collection_name="cxr_images",
                    ids=img_ids,
                    vectors=image_embeddings,
                    payloads=img_payloads
                )

    print("--- Phase 3: Indexing Complete ---")

if __name__ == "__main__":
    main()
