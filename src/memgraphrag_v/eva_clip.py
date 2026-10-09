"""EVA-CLIP encoder, reproducing MG2's ``EvaClipModel`` preprocessing.

Same as MG2 (``src/mmgraphrag/embedding_model/eva_clip.py``): CLIP tokenizer
from the checkpoint with pad = eos, ``CLIPImageProcessor`` from the
openai/clip-vit-large-patch14 config, fp16 weights on CUDA, the model's own
``encode_image`` / ``encode_text`` under autocast, and L2-normalised outputs.

Different from MG2: this class takes already-opened images, so it never
replaces an unreadable file with a black image (``embeddings.py`` records the
failure instead), and it reports which texts were truncated.

torch and transformers are imported here only, so the rest of the package
installs without them (``pip install -e ".[eva]"`` on the GPU node).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class EvaClip:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        processor_dir: str | Path,
        device: str = "cuda",
        batch_size: int = 16,
        max_text_tokens: int = 77,
    ) -> None:
        import torch
        from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_text_tokens = max_text_tokens

        self.tokenizer = CLIPTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = CLIPImageProcessor.from_pretrained(str(processor_dir), trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = (
            AutoModel.from_pretrained(str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        self.dim: int = self.model.config.projection_dim

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """(n, dim) float32, L2-normalised. Images must already be RGB."""
        torch = self._torch

        def encode(batch: list[Image.Image]):
            pixels = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            if self.model.dtype == torch.float16:
                pixels = pixels.half()
            return self.model.encode_image(pixels)

        return self._run(images, self.batch_size, encode)

    def encode_texts(self, texts: list[str]) -> tuple[np.ndarray, list[bool]]:
        """(n, dim) float32, L2-normalised, plus which texts were cut to
        ``max_text_tokens`` tokens."""
        lengths = [len(ids) for ids in self.tokenizer(texts)["input_ids"]]
        truncated = [n > self.max_text_tokens for n in lengths]

        def encode(batch: list[str]):
            ids = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_text_tokens,
            ).input_ids.to(self.device)
            return self.model.encode_text(ids)

        # MG2 uses 8x the image batch size for text.
        return self._run(texts, self.batch_size * 8, encode), truncated

    def _run(self, items: list, batch_size: int, encode) -> np.ndarray:
        torch = self._torch
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks = []
        autocast = torch.autocast("cuda", enabled=self.device.startswith("cuda"))
        with torch.no_grad(), autocast:
            for i in range(0, len(items), batch_size):
                chunks.append(encode(items[i : i + batch_size]).float().cpu())
        vectors = torch.cat(chunks)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.numpy()
