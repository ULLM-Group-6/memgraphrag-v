"""Download only the HF-format EVA checkpoint and processor required by MG2."""
import json
from pathlib import Path
from huggingface_hub import model_info, snapshot_download, hf_hub_download

root = Path('checkpoints/eva-smoke')
root.mkdir(parents=True, exist_ok=True)
record = root / 'revisions.json'
if record.exists():
    revisions = json.loads(record.read_text())
else:
    revisions = {
        'eva_repo': 'BAAI/EVA-CLIP-8B',
        'eva_revision': model_info('BAAI/EVA-CLIP-8B').sha,
        'processor_repo': 'openai/clip-vit-large-patch14',
        'processor_revision': model_info('openai/clip-vit-large-patch14').sha,
    }
    record.write_text(json.dumps(revisions, indent=2))

snapshot_download(
    repo_id=revisions['eva_repo'], revision=revisions['eva_revision'],
    local_dir=root / 'EVA-CLIP-8B',
    allow_patterns=['*.json', '*.py', 'merges.txt', 'vocab.json',
                    'pytorch_model-*.bin'],
    max_workers=2,
)
hf_hub_download(
    repo_id=revisions['processor_repo'], revision=revisions['processor_revision'],
    filename='preprocessor_config.json', local_dir=root / 'clip-processor',
)
print('Downloads complete. Pinned revisions:', json.dumps(revisions, indent=2))
