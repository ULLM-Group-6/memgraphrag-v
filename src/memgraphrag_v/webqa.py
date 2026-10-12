"""Misha's WebQA subset -> corpus images and manifests.

Copy the builder's files (``dataset`` branch) into ``$MGRV_ARTIFACTS/webqa/``,
then run from the repo:

    python -m memgraphrag_v.webqa download    # needs internet: login node or staging partition
    python -m memgraphrag_v.webqa manifests   # offline; re-run whenever webqa/ or the images change

``download`` fetches every image in ``images_to_download.jsonl`` into
``corpus/images/<image_id>.<ext>``. It is resumable: images already on disk
are skipped. Wikimedia only serves thumbnails at its standard widths
(https://w.wiki/GHai) and WebQA's URLs use other widths, so a thumbnail URL is
rewritten to the next standard width, and the original file is fetched if
that fails. Each finished attempt is appended to ``corpus/download.jsonl``.

``manifests`` writes ``manifests/{text_manifest,image_manifest,questions}.jsonl``:

- passages are copied as they are;
- images gain ``path``, ``width``, ``height`` and ``ok`` (the file opens);
- questions from ``build_qa.json`` become split ``dev`` (the graph-building
  questions, used to tune beta) and ``test_qa.json`` becomes split ``test``.
  Snippet IDs get the ``txt_`` prefix their passages have, gold documents are
  the titles of the gold sources, and questions sharing gold evidence share a
  ``group_id``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from .io import read_jsonl, write_jsonl
from .paths import Artifacts
from .schemas import ImageRecord, PassageRecord, QuestionRecord

USER_AGENT = "MemGraphRAG-V research downloader (https://github.com/ULLM-Group-6/memgraphrag-v)"
THUMB_STEPS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)
"""Wikimedia's standard thumbnail widths; other widths are rejected (HTTP 400)."""

_THUMB = re.compile(r"^(?P<base>.+)/thumb(?P<file>/[0-9a-f]/[0-9a-f]{2}/[^/]+)/(?P<width>\d+)px-(?P<name>[^/]+)$")

Fetch = Callable[[str], bytes]


# --- Download ----------------------------------------------------------------


def candidate_urls(url: str) -> list[str]:
    """URLs to try for one image, best first: the thumbnail at the next
    standard width, then the original file. Non-thumbnail URLs are kept."""
    m = _THUMB.match(url)
    if not m:
        return [url]
    width = int(m["width"])
    step = next((s for s in THUMB_STEPS if s >= width), THUMB_STEPS[-1])
    return [f"{m['base']}/thumb{m['file']}/{step}px-{m['name']}", f"{m['base']}{m['file']}"]


def fetch_url(url: str, retries: int = 4, rate_limit_retries: int = 20, timeout: float = 60) -> bytes:
    """GET with a descriptive User-Agent, as Wikimedia asks. Waits out rate
    limits (429, honouring Retry-After) up to ``rate_limit_retries`` times and
    retries server and network errors with backoff; other errors raise."""
    failures = limited = 0
    while True:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and limited < rate_limit_retries:
                limited += 1
                wait = e.headers.get("Retry-After")
                time.sleep(min(float(wait), 120) if wait and wait.isdigit() else 15)
            elif e.code >= 500 and failures < retries:
                failures += 1
                time.sleep(2**failures)
            else:
                raise
        except (urllib.error.URLError, TimeoutError):
            if failures == retries:
                raise
            failures += 1
            time.sleep(2**failures)


def local_file(images_dir: Path, image_id: str) -> Path | None:
    """The downloaded file for ``image_id``, if any."""
    return next((p for p in sorted(images_dir.glob(f"{image_id}.*")) if p.suffix != ".part"), None)


def download_images(
    rows: Iterable[dict],
    images_dir: Path,
    log_path: Path,
    fetch: Fetch = fetch_url,
    delay: float = 0.1,
) -> Counter:
    """Download every row's ``imgUrl`` to ``images_dir/<image_id><ext>``,
    skipping images already there. Returns counts of skipped/ok/failed."""
    images_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for row in rows:
            image_id = str(row["image_id"])
            if local_file(images_dir, image_id):
                counts["skipped"] += 1
                continue
            entry = {"image_id": image_id, "status": "failed", "url": None, "bytes": 0, "errors": []}
            for url in candidate_urls(row["imgUrl"]):
                try:
                    data = fetch(url)
                except Exception as e:
                    entry["errors"].append(f"{url}: {type(e).__name__}: {e}")
                    continue
                suffix = Path(urllib.parse.unquote(url.rsplit("/", 1)[-1])).suffix.lower() or ".img"
                part = images_dir / f"{image_id}{suffix}.part"
                part.write_bytes(data)
                part.replace(images_dir / f"{image_id}{suffix}")
                entry.update(status="ok", url=url, bytes=len(data))
                break
            counts[entry["status"]] += 1
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            if delay:
                time.sleep(delay)
    return counts


# --- Manifests ---------------------------------------------------------------


def image_records(rows: Iterable[dict], images_dir: Path) -> list[ImageRecord]:
    """Builder image rows plus the local file, its size and whether it opens.
    A missing file gets the path it would have and ``ok`` false."""
    records = []
    for row in rows:
        file = local_file(images_dir, row["image_id"])
        size = _image_size(file) if file else None
        records.append(ImageRecord(
            **row,
            path=file.name if file else f"{row['image_id']}.missing",
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


def check_references(
    passages: list[PassageRecord], images: list[ImageRecord], questions: list[QuestionRecord]
) -> dict[str, int]:
    """Raise if IDs repeat or a caption or gold source is missing from the
    corpus; return counts worth reading before a run."""
    passage_ids = {p.passage_id for p in passages}
    image_ids = {i.image_id for i in images}
    for name, ids, n in (("passage", passage_ids, len(passages)), ("image", image_ids, len(images)),
                         ("question", {q.qid for q in questions}, len(questions))):
        if len(ids) != n:
            raise ValueError(f"repeated {name} IDs")
    problems = [f"image {i.image_id}: caption {p} missing" for i in images for p in i.passage_ids if p not in passage_ids]
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
    }


def write_manifests(artifacts: Artifacts) -> dict[str, int]:
    source = artifacts.webqa_dir
    passages = read_jsonl(source / "text_manifest.jsonl", PassageRecord)
    with open(source / "image_manifest.jsonl", encoding="utf-8") as f:
        images = image_records((json.loads(line) for line in f if line.strip()), artifacts.images_dir)
    load = lambda name: json.loads((source / name).read_text(encoding="utf-8"))
    questions = question_records(load("build_qa.json"), load("test_qa.json"))
    counts = check_references(passages, images, questions)
    write_jsonl(artifacts.text_manifest, passages)
    write_jsonl(artifacts.image_manifest, images)
    write_jsonl(artifacts.questions, questions)
    return counts


# --- Command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from .config import ExperimentConfig

    parser = argparse.ArgumentParser(description="Download WebQA images and write the manifests.")
    parser.add_argument("step", choices=["download", "manifests"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, help="download only the first N images (smoke runs)")
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between downloads")
    args = parser.parse_args(argv)

    artifacts = Artifacts.from_env(ExperimentConfig.load(args.config).artifacts_root)
    if args.step == "download":
        with open(artifacts.webqa_dir / "images_to_download.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()][: args.limit]
        artifacts.download_log.parent.mkdir(parents=True, exist_ok=True)
        counts = download_images(rows, artifacts.images_dir, artifacts.download_log, delay=args.delay)
        print(json.dumps({"images_dir": str(artifacts.images_dir), **counts}, indent=2))
    else:
        print(json.dumps({"manifests": str(artifacts.text_manifest.parent), **write_manifests(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
