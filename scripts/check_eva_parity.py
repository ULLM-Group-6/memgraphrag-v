"""Check that our EvaClip adapter gives the same vectors as MG2's EvaClipModel.

Encodes MG2's impala demo images and a few texts with both, on one GPU, and
fails if any vector differs by more than --atol. MG2's adapter imports a few
packages ours does not need, so install them first:
    pip install rich openai tenacity
Then, inside a GPU job:
    python scripts/check_eva_parity.py --mg2 external/MG2-RAG
"""

import argparse
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.eva_clip import EvaClip
from memgraphrag_v.paths import Artifacts

TEXTS = [
    "a photograph of an impala antelope",
    "a photograph of an airplane",
    "a line chart showing revenue over time",
]


def import_mg2(repo: Path):
    # Skip MG2's package initializer, which imports its graph, SAM3 and CuPy
    # modules; load only the unmodified EVA adapter and its helpers.
    package = types.ModuleType("mg2_parity")
    package.__path__ = [str(repo / "src" / "mmgraphrag")]
    sys.modules[package.__name__] = package
    config_utils = importlib.import_module("mg2_parity.utils.config_utils")
    eva_clip = importlib.import_module("mg2_parity.embedding_model.eva_clip")
    return config_utils, eva_clip


def load_mg2_encoder(mg2, checkpoint: Path, processor: Path):
    config_utils, eva_clip = mg2
    config = config_utils.BaseConfig(
        multimodal_embedding_model_name=str(checkpoint),
        clip_image_processor_name=str(processor),
        clip_device=0,
        multimodal_embedding_batch_size=16,
        multimodal_embedding_max_model_len=77,
    )
    return eva_clip.EvaClipModel(global_config=config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mg2", type=Path, required=True, help="MG2-RAG checkout")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()

    mg2 = import_mg2(args.mg2)  # before loading any model, so a missing package fails fast
    enc = ExperimentConfig.load(args.config).visual_encoder
    models = Artifacts.from_env().models_dir
    checkpoint = models / enc.name.rsplit("/", 1)[-1]
    processor = models / enc.processor.rsplit("/", 1)[-1]
    demo = args.mg2 / "examples" / "data" / "impala_demo"
    images = [Image.open(demo / name).convert("RGB") for name in ("kb_image.jpg", "query_image.jpg")]

    ours = EvaClip(checkpoint, processor, batch_size=enc.batch_size, max_text_tokens=enc.max_text_tokens)
    our_images = ours.encode_images(images)
    our_texts, _ = ours.encode_texts(TEXTS)
    del ours  # free the GPU before loading the second copy of the 8B model
    torch.cuda.empty_cache()

    theirs = load_mg2_encoder(mg2, checkpoint, processor)
    mg2_images = theirs.batch_encode(images=images)
    mg2_texts = theirs.batch_encode(texts=TEXTS)

    for name, a, b in [("images", our_images, mg2_images), ("texts", our_texts, mg2_texts)]:
        diff = float(np.max(np.abs(a - b)))
        cosine = float(np.min(np.sum(a * b, axis=1)))
        print(f"{name}: max |diff| = {diff:.2e}, min cosine = {cosine:.6f}")
        if diff > args.atol:
            raise SystemExit(f"{name}: our vectors differ from MG2's by {diff:.2e} > {args.atol}")
    print("parity OK")


if __name__ == "__main__":
    main()
