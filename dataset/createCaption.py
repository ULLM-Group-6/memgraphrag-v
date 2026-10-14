import os
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

manifest_path = "image_manifest.jsonl"
output_manifest_path = "caption_manifest.jsonl"

PROMPT = """Describe only what can be directly observed in the image.
Describe this image so that someone who cannot see it could later answer factual questions about it. Be objective and specific. Name the main subject and its category; if you can confidently identify the specific entity (species, breed, material, landmark, object type), name it, otherwise describe it without guessing. Mention distinctive visible attributes (colours, shape, markings, setting, number of objects) and transcribe any legible text. 
Do not identify objects using external knowledge. Do not infer the object's historical name, origin, purpose, creator, inventor, or other facts that are not visually observable. 
Return a concise factual visual description. Write 3-5 sentences of plain prose."""

print("Loading Qwen2.5-VL model from local cache...")
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
cache_directory = "/scratch-shared/scur0462/hf_cache"

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_directory,
    local_files_only=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_directory,
    local_files_only=True
)

print(f"Reading from {manifest_path} and writing to {output_manifest_path}...")

with open(manifest_path, "r", encoding="utf-8") as infile, \
     open(output_manifest_path, "w", encoding="utf-8") as outfile:

    for line in infile:
        if not line.strip():
            continue

        data = json.loads(line)
        img_id = data["image_id"]
        img_path = f"images/{img_id}.jpg"

        if not os.path.exists(img_path):
            print(f"Skipping {img_id}: Image not found at {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=150)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        new_data = {
            "image_id": img_id,
            "doc_id": data.get("doc_id", ""),
            "passage_ids": data.get("passage_ids", [f"cap_{img_id}"]),
            "source_url": data.get("source_url", ""),
            "commons_url": data.get("commons_url", ""),
            "image_title": data.get("doc_id", ""),
            "generated_description": caption.strip()
        }

        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        outfile.flush()

print(f"Finished generating captions. Output saved to {output_manifest_path}")
