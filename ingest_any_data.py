import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Universal Multimodal Ingestor for Actian Copilot")
    parser.add_argument("--csv", required=True, help="Path to your metadata CSV file")
    parser.add_argument("--img_col", required=True, help="Column name containing image filenames")
    parser.add_argument("--text_col", required=True, help="Column name containing clinical descriptions/labels")
    parser.add_argument("--img_dir", required=True, help="Directory where images are stored")
    
    args = parser.parse_args()

    print(f"--- 🔄 Ingesting {args.csv} ---")
    df = pd.read_csv(args.csv)

    # 1. Map paths and verify existence
    def verify_and_map(img_id):
        # Handle cases where extension might be missing in CSV
        if not str(img_id).lower().endswith(('.png', '.jpg', '.jpeg')):
            img_id = f"{img_id}.png" # Default to png
            
        full_path = os.path.join(args.img_dir, img_id)
        return full_path if os.path.exists(full_path) else None

    df['image_paths'] = df[args.img_col].apply(verify_and_map)
    initial_count = len(df)
    df = df.dropna(subset=['image_paths'])
    
    print(f"Verified {len(df)}/{initial_count} images found on disk.")

    # 2. Standardize columns for our pipeline
    # Rename the text column to 'findings' (our pipeline's internal name)
    df = df.rename(columns={args.text_col: 'findings'})
    
    # Add a dummy 'xml_file' or 'id' column for reference
    df['xml_file'] = df[args.img_col]
    
    # Save for indexing
    df.to_csv("processed_data.csv", index=False)
    print("✅ Ingestion Complete. Result saved to 'processed_data.csv'")

if __name__ == "__main__":
    main()
