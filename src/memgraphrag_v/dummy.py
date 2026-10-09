"""A tiny synthetic corpus in the MMQA manifest format.

It lets us run image retrieval end to end before the real dataset is ready.
Each of the twelve documents "<Colour> <shape>" (e.g. "Red circle") has one
picture of that shape and one passage, and each has the question "Which
picture shows a red circle?". A working encoder should rank the gold
document first. Edge cases included:

- "Red circle" has a second, larger picture (one image per document);
- "Grey frame" has a picture but no passages;
- "Toy corpus" has passages but no picture;
- "Broken picture" has an image file that cannot be read (``ok`` false).

    python -m memgraphrag_v.dummy [ROOT]    # default: $MGRV_ARTIFACTS
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from .io import write_jsonl
from .paths import Artifacts
from .schemas import ImageRecord, PassageRecord, QuestionRecord

COLOURS = {"red": (220, 30, 30), "green": (30, 160, 60), "blue": (30, 60, 220), "yellow": (240, 200, 20)}
SHAPES = ("circle", "square", "triangle")
SIZE = 224


def make_dummy_dataset(artifacts: Artifacts) -> dict[str, int]:
    """Write manifests and images under ``artifacts.root``; returns counts."""
    artifacts.images_dir.mkdir(parents=True, exist_ok=True)
    passages: list[PassageRecord] = []
    images: list[ImageRecord] = []
    questions: list[QuestionRecord] = []

    def add_passage(doc_id: str, text: str) -> str:
        passage = PassageRecord(doc_id=doc_id, passage_id=_hex(doc_id, text), text=text)
        passages.append(passage)
        return passage.passage_id

    def add_image(doc_id: str, picture: Image.Image | None, passage_ids: list[str], n: int = 0) -> ImageRecord:
        image_id = _hex(doc_id, f"image {n}")
        path = f"{image_id}.png"
        if picture is None:
            (artifacts.images_dir / path).write_bytes(b"not an image")
        else:
            picture.save(artifacts.images_dir / path)
        image = ImageRecord(
            image_id=image_id, doc_id=doc_id, path=path, passage_ids=passage_ids, ok=picture is not None,
            width=picture.width if picture else None, height=picture.height if picture else None,
        )
        images.append(image)
        return image

    for colour, rgb in COLOURS.items():
        for shape in SHAPES:
            doc_id = f"{colour.capitalize()} {shape}"
            passage_id = add_passage(doc_id, f"{doc_id} is one of the figures in the toy corpus.")
            image = add_image(doc_id, _draw(shape, rgb, margin=60), [passage_id])
            if doc_id == "Red circle":
                add_image(doc_id, _draw(shape, rgb, margin=20), [passage_id], n=1)
            questions.append(QuestionRecord(
                qid=_hex("question", doc_id), question=f"Which picture shows a {colour} {shape}?",
                answers=[doc_id], gold_doc_ids=[doc_id], gold_image_ids=[image.image_id],
                gold_passage_ids=[], distractor_image_ids=[], distractor_passage_ids=[],
                q_type="ImageQ", modalities=["image"], split="dev" if len(questions) % 2 == 0 else "test",
                group_id=len(questions),
            ))

    add_image("Grey frame", _draw("square", (128, 128, 128), margin=40, fill=False), [])
    add_passage("Toy corpus", "The toy corpus has coloured shapes on a white background.")
    add_passage("Toy corpus", "Every shape is drawn once, except the red circle.")
    add_image("Broken picture", None, [add_passage("Broken picture", "This picture cannot be opened.")])

    write_jsonl(artifacts.text_manifest, passages)
    write_jsonl(artifacts.image_manifest, images)
    write_jsonl(artifacts.questions, questions)
    return {"passages": len(passages), "images": len(images), "questions": len(questions)}


def _draw(shape: str, rgb: tuple[int, int, int], margin: int, fill: bool = True) -> Image.Image:
    picture = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(picture)
    box = (margin, margin, SIZE - margin, SIZE - margin)
    style = {"fill": rgb} if fill else {"outline": rgb, "width": 8}
    if shape == "circle":
        draw.ellipse(box, **style)
    elif shape == "square":
        draw.rectangle(box, **style)
    else:
        draw.polygon([(SIZE // 2, margin), (SIZE - margin, SIZE - margin), (margin, SIZE - margin)], **style)
    return picture


def _hex(*parts: str) -> str:
    """A stable 32-character hex ID, like MMQA's."""
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the synthetic dummy dataset.")
    parser.add_argument("root", nargs="?", help="artifacts root (default: $MGRV_ARTIFACTS)")
    args = parser.parse_args(argv)
    artifacts = Artifacts.from_env(args.root)
    print(artifacts.root, make_dummy_dataset(artifacts))


if __name__ == "__main__":
    main()
