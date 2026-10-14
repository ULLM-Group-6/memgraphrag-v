"""Our WebQA subset -> corpus images and manifests.

Copy the builder's files (``dataset`` branch) into ``$MGRV_ARTIFACTS/webqa/``,
then run from the repo:

    python -m memgraphrag_v.webqa images --tsv <path to shortlisted_imgs.tsv>
    python -m memgraphrag_v.webqa manifests   # re-run whenever webqa/ or the images change

``images`` unpacks the corpus images from the TSV cut out of WebQA's
``imgs.tsv`` (one ``<image_id> TAB <base64 bytes>`` line per image) into
``corpus/images/<image_id>.jpg``, the same names ``unpack_tsv.py`` on the
``dataset`` branch uses. Should a file not really be a JPEG, Pillow still
reads it by its content. Images already on disk are skipped.

``manifests`` writes ``manifests/{text_manifest,image_manifest,questions,captions}.jsonl``:

- passages are copied as they are;
- images gain ``path``, ``width``, ``height`` and ``ok`` (the file opens);
- questions from ``build_qa.json`` become split ``dev`` (the graph-building
  questions, used to tune beta) and ``test_qa.json`` becomes split ``test``.
  Snippet IDs get the ``txt_`` prefix their passages have, gold documents are
  the titles of the gold sources, and questions sharing gold evidence share a
  ``group_id``;
- captions are the Qwen2.5-VL descriptions in ``caption_manifest.jsonl``,
  one per image.
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image

from .io import read_jsonl, write_jsonl
from .paths import Artifacts
from .schemas import CaptionRecord, ImageRecord, PassageRecord, QuestionRecord

CAPTION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CAPTION_PROMPT_VERSION = "createCaption.py@2c94363"
"""The caption script on the dataset branch, at the commit that made the captions."""


# --- Images ------------------------------------------------------------------


def image_file(images_dir: Path, image_id: str) -> Path:
    return images_dir / f"{image_id}.jpg"


def unpack_images(tsv_path: Path, images_dir: Path) -> Counter:
    """Write every image in the TSV to ``images_dir``, skipping files that are
    already there. Returns counts of written and skipped images."""
    images_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            image_id, data = line.strip().split("\t")
            out = image_file(images_dir, image_id)
            if out.exists():
                counts["skipped"] += 1
                continue
            # write to a temporary name first, so a killed run leaves no half-written image
            part = out.with_suffix(".part")
            part.write_bytes(base64.b64decode(data))
            part.replace(out)
            counts["written"] += 1
    return counts


# --- Manifests ---------------------------------------------------------------


def image_records(rows: Iterable[dict], images_dir: Path) -> list[ImageRecord]:
    """Builder image rows plus the local file, its size and whether it opens.
    A missing file keeps the path it would have and gets ``ok`` false."""
    records = []
    for row in rows:
        file = image_file(images_dir, row["image_id"])
        size = _image_size(file)
        records.append(ImageRecord(
            **row,
            path=file.name,
            width=size[0] if size else None,
            height=size[1] if size else None,
            ok=size is not None,
        ))
    return records


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.convert("RGB")
            return image.size
    except Exception:
        return None


MODALITIES = {"image": ["image"], "text": ["text"], "both": ["image", "text"]}


def question_records(build: list[dict], test: list[dict]) -> list[QuestionRecord]:
    """Builder questions in our schema: build -> dev, test -> test."""
    rows = [(q, "dev") for q in build] + [(q, "test") for q in test]
    groups = _evidence_groups([q for q, _ in rows])
    return [
        QuestionRecord(
            qid=q["qid"],
            question=q["question"],
            answers=q["answers"],
            gold_doc_ids=list(dict.fromkeys(d["title"] for d in q["gold_documents"])),
            gold_image_ids=q["gold_image_ids"],
            gold_passage_ids=[f"txt_{s}" for s in q["gold_snippet_ids"]],
            distractor_image_ids=q["distractor_image_ids"],
            distractor_passage_ids=[f"txt_{s}" for s in q["distractor_snippet_ids"]],
            q_type=q.get("Qcate"),
            topic=q.get("topic"),
            modalities=MODALITIES[q["evidence_class"]],
            split=split,
            group_id=group,
        )
        for (q, split), group in zip(rows, groups)
    ]


def _evidence_groups(questions: list[dict]) -> list[int]:
    """Connected components of questions linked by a shared gold source,
    numbered in order of first appearance."""
    parent = list(range(len(questions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, q in enumerate(questions):
        for source in [f"img:{s}" for s in q["gold_image_ids"]] + [f"txt:{s}" for s in q["gold_snippet_ids"]]:
            if source in first:
                parent[find(i)] = find(first[source])
            else:
                first[source] = i
    numbers: dict[int, int] = {}
    return [numbers.setdefault(find(i), len(numbers)) for i in range(len(questions))]


def caption_records(rows: Iterable[dict]) -> list[CaptionRecord]:
    """The generated descriptions in our schema. The source rows also repeat the
    image's title and URLs; those are already in the image manifest."""
    return [
        CaptionRecord(
            image_id=row["image_id"],
            text=row["generated_description"],
            model=CAPTION_MODEL,
            prompt_version=CAPTION_PROMPT_VERSION,
        )
        for row in rows
    ]


def check_references(
    passages: list[PassageRecord],
    images: list[ImageRecord],
    questions: list[QuestionRecord],
    captions: list[CaptionRecord],
) -> dict[str, int]:
    """Raise if IDs repeat, an image's caption passage or a gold source is
    missing from the corpus, or a generated caption belongs to no image;
    return counts worth reading before a run."""
    passage_ids = {p.passage_id for p in passages}
    image_ids = {i.image_id for i in images}
    captioned = {c.image_id for c in captions}
    for name, ids, n in (("passage", passage_ids, len(passages)), ("image", image_ids, len(images)),
                         ("question", {q.qid for q in questions}, len(questions)),
                         ("generated caption", captioned, len(captions))):
        if len(ids) != n:
            raise ValueError(f"repeated {name} IDs")
    problems = [f"image {i.image_id}: caption {p} missing" for i in images for p in i.passage_ids if p not in passage_ids]
    problems += [f"generated caption for unknown image {c}" for c in sorted(captioned - image_ids)]
    for q in questions:
        problems += [f"question {q.qid}: gold image {g} not in corpus" for g in q.gold_image_ids if g not in image_ids]
        problems += [f"question {q.qid}: gold passage {g} not in corpus" for g in q.gold_passage_ids
                     if g not in passage_ids]
    if problems:
        raise ValueError(f"{len(problems)} broken references, e.g. " + "; ".join(problems[:5]))
    ok = {i.image_id for i in images if i.ok}
    gold_images = {g for q in questions for g in q.gold_image_ids}
    return {
        "passages": len(passages),
        "images": len(images),
        "images_ok": len(ok),
        "questions_dev": sum(q.split == "dev" for q in questions),
        "questions_test": sum(q.split == "test" for q in questions),
        "groups": len({q.group_id for q in questions}),
        "gold_images": len(gold_images),
        "gold_images_not_ok": len(gold_images - ok),
        "generated_captions": len(captions),
        "images_without_generated_caption": len(image_ids - captioned),
    }


def write_manifests(artifacts: Artifacts) -> dict[str, int]:
    source = artifacts.webqa_dir
    passages = read_jsonl(source / "text_manifest.jsonl", PassageRecord)
    with open(source / "image_manifest.jsonl", encoding="utf-8") as f:
        images = image_records((json.loads(line) for line in f if line.strip()), artifacts.images_dir)
    load = lambda name: json.loads((source / name).read_text(encoding="utf-8"))
    questions = question_records(load("build_qa.json"), load("test_qa.json"))
    with open(source / "caption_manifest.jsonl", encoding="utf-8") as f:
        captions = caption_records(json.loads(line) for line in f if line.strip())
    counts = check_references(passages, images, questions, captions)
    write_jsonl(artifacts.text_manifest, passages)
    write_jsonl(artifacts.image_manifest, images)
    write_jsonl(artifacts.questions, questions)
    write_jsonl(artifacts.captions, captions)
    return counts


# --- Command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from .config import ExperimentConfig

    parser = argparse.ArgumentParser(description="Unpack the WebQA images and write the manifests.")
    parser.add_argument("step", choices=["images", "manifests"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tsv", type=Path, help="images step: the shortlisted_imgs.tsv from the dataset work")
    args = parser.parse_args(argv)

    artifacts = Artifacts.from_env(ExperimentConfig.load(args.config).artifacts_root)
    if args.step == "images":
        if args.tsv is None:
            parser.error("the images step needs --tsv")
        counts = unpack_images(args.tsv, artifacts.images_dir)
        print(json.dumps({"images_dir": str(artifacts.images_dir), **counts}, indent=2))
    else:
        print(json.dumps({"manifests": str(artifacts.text_manifest.parent), **write_manifests(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
