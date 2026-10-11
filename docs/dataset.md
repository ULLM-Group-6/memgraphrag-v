text_manifest.jsonl Actual caption text is stored in this file eg {
  "doc_id": "Center (gridiron football) - Wikipedia",
  "passage_id": "txt_d5df4c700dba11ecb1e81171463288e9_9",
  "text": "The center is the innermost lineman ...",
  "source_url": "https://en.wikipedia.org/wiki/Center_(American_football)"
}


image_manifest.jsonl to identify image nodes and their metadata. eg {
  "doc_id": "Bandol rose",
  "passage_id": "cap_30388563", ID of the associated caption passage.
  "text": "Caption text for the image...",
  "source_url": "https://commons.wikimedia.org/..."
}


images_to_download.jsonl to download the corpus images, then process those local image files.


build_qa.json to inspect the graph-building questions and their evidence.


test_qa.json only for evaluation
