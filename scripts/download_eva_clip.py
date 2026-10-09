"""Download the EVA-CLIP checkpoint and image-processor config pinned in the
experiment config into $MGRV_ARTIFACTS/models/.

Run on a login node (needs internet, about 30 GB):
    python scripts/download_eva_clip.py [--config configs/default.yaml]
"""

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.paths import Artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    enc = config.visual_encoder
    models = Artifacts.from_env(config.artifacts_root).models_dir

    # Only the HF-format weights and the remote code MG2 loads.
    snapshot_download(
        repo_id=enc.name,
        revision=enc.revision,
        local_dir=models / enc.name.rsplit("/", 1)[-1],
        allow_patterns=["*.json", "*.py", "merges.txt", "vocab.json", "pytorch_model-*.bin"],
        max_workers=2,
    )
    hf_hub_download(
        repo_id=enc.processor,
        revision=enc.processor_revision,
        filename="preprocessor_config.json",
        local_dir=models / enc.processor.rsplit("/", 1)[-1],
    )
    print(f"Downloaded {enc.name}@{enc.revision} and {enc.processor}@{enc.processor_revision} to {models}")


if __name__ == "__main__":
    main()
