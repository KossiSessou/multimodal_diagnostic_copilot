import os
import pandas as pd
from tqdm import tqdm

def ingest_ham10000(csv_path, img_dir):
    print(f"--- 🔄 Ingesting HAM10000 Data ---")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Metadata not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 1. Diagnostic Mapping
    dx_map = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesion'
    }

    # 2. Create Clinical Findings
    def create_findings(row):
        age = f"{int(row['age'])} year old" if pd.notna(row['age']) else "Unknown age"
        sex = row['sex'] if pd.notna(row['sex']) else "unknown sex"
        loc = row['localization'] if pd.notna(row['localization']) else "unknown location"
        diag = dx_map.get(row['dx'], "skin lesion")
        
        return f"Patient is a {age} {sex}. Lesion located on the {loc}. Clinical diagnosis: {diag}."

    print("Generating clinical descriptions...")
    df['findings'] = df.apply(create_findings, axis=1)
    df['xml_file'] = df['image_id']

    # 3. Map Image Paths
    print("Verifying image files...")
    image_paths = []
    for img_id in tqdm(df['image_id']):
        path = os.path.join(img_dir, f"{img_id}.jpg")
        if os.path.exists(path):
            image_paths.append(path)
        else:
            image_paths.append(None)
    
    df['image_paths'] = image_paths
    df = df.dropna(subset=['image_paths'])
    
    # 4. Save for pipeline
    df.to_csv("processed_data.csv", index=False)
    print(f"✅ Successfully processed {len(df)} images.")
    print("Next step: Run 'python index_data.py'")

if __name__ == "__main__":
    ingest_ham10000("data/HAM10000_metadata.csv", "data/HAM10000_images/")
