import base64
from pathlib import Path

# Define input and output paths
tsv_path = Path("/scratch-shared/scur0462/WebQA/shortlisted_imgs.tsv")
output_dir = Path.home() / "memgraphrag-v/images"

# Create the images directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

count = 0
with open(tsv_path, "r", encoding="utf-8") as f:
    for line in f:
        # Split the tab-separated values
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
            
        img_id, b64_data = parts
        
        # Decode the Base64 string to raw image bytes
        img_bytes = base64.b64decode(b64_data)
        out_file = output_dir / f"{img_id}.jpg"
        
        # Write the bytes to a .jpg file
        with open(out_file, "wb") as img_file:
            img_file.write(img_bytes)
            
        count += 1
        if count % 500 == 0:
            print(f"Unpacked {count} images...")

print(f"Done! {count} images successfully saved to {output_dir}")
