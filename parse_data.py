import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def parse_openi_xml(reports_dir, images_dir):
    """
    Parses OpenI XML files to extract clinical findings, impressions, and image mappings.
    """
    data_list = []
    
    # Iterate through XML files in the reports directory
    for root_dir, _, files in os.walk(reports_dir):
        for file in tqdm(files, desc="Parsing XMLs"):
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # Extract Findings and Impression
                    findings = ""
                    impression = ""
                    
                    # OpenI XML structure typically has MedlineCitation -> Article -> Abstract -> AbstractText
                    for abstract_text in root.findall(".//AbstractText"):
                        label = abstract_text.get('Label') # Note: Case sensitivity (Label vs label)
                        if not label:
                            label = abstract_text.get('label')
                            
                        if label and label.upper() == 'FINDINGS':
                            findings = abstract_text.text if abstract_text.text else ""
                        elif label and label.upper() == 'IMPRESSION':
                            impression = abstract_text.text if abstract_text.text else ""
                    
                    # Extract Image IDs (parentImage id)
                    images = []
                    for parent_image in root.findall(".//parentImage"):
                        img_id = parent_image.get('id')
                        if img_id:
                            # Map to actual file path
                            img_filename = f"{img_id}.png"
                            
                            # Search for the image file in the images_dir
                            # Sometimes images are also in subfolders
                            found = False
                            for img_root, _, img_files in os.walk(images_dir):
                                if img_filename in img_files:
                                    images.append(os.path.join(img_root, img_filename))
                                    found = True
                                    break
                    
                    # Add to data list if we have text OR images (to be more inclusive)
                    if findings or impression:
                        data_list.append({
                            "xml_file": file,
                            "findings": findings,
                            "impression": impression,
                            "image_paths": images
                        })
                        
                except Exception as e:
                    print(f"Error parsing {file}: {e}")
                    
    return data_list

if __name__ == "__main__":
    REPORTS_PATH = "data/reports"
    IMAGES_PATH = "data/images"
    
    if not os.path.exists(REPORTS_PATH):
        print(f"Error: Reports directory {REPORTS_PATH} not found. Run download_data.sh first.")
        exit(1)
        
    print("--- Starting XML Parsing ---")
    processed_data = parse_openi_xml(REPORTS_PATH, IMAGES_PATH)
    
    if not processed_data:
        print("Error: No data parsed from XML files. Please check the reports directory structure.")
        exit(1)

    # Create a DataFrame for easy handling
    df = pd.DataFrame(processed_data)
    
    # Explode the image_paths to have one row per image (standard for multimodal RAG)
    # Check if image_paths column exists and has lists
    if "image_paths" in df.columns:
        df_exploded = df.explode("image_paths").dropna(subset=["image_paths"])
        if df_exploded.empty:
            print("Warning: No images matched with reports. Using text-only data.")
            df_exploded = df
    else:
        df_exploded = df
    
    # Save the mapping
    output_file = "processed_data.csv"
    df_exploded.to_csv(output_file, index=False)
    
    print(f"--- Parsing Complete ---")
    print(f"Total reports parsed: {len(df)}")
    print(f"Total image-report pairs: {len(df_exploded)}")
    print(f"Mapping saved to: {output_file}")
