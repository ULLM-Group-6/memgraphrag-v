#!/usr/bin/env python3
"""
Validate the manifests written by build_mmqa_manifests.py.  Exit code 1 if any ERROR is found.

  python check_manifests.py --manifest-dir manifests --images-dir images/final_dataset_images --show 3

Checks
  1  files exist; line counts and sha256 match report.json
  2  schema: required keys and types in every row
  3  uniqueness of passage_id / image_id / qid
  4  referential integrity: every id used by an image or question exists in the corpus,
     gold and distractor sets are disjoint, gold docs exist
  5  image files exist, open, and match the manifest's size/ok flag (skip with --skip-image-open)
  6  split hygiene: sizes, no qid in two splits, no gold document shared by dev and test
  7  leakage: index-side files (text/image manifests) contain no question/answer/gold fields
  8  warnings: images with no same-title passage, questions that mention "this image", empties
"""
import argparse
import hashlib
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

STR, NONE = str, type(None)
SCHEMA = {
    "text_manifest.jsonl": {"doc_id": STR, "passage_id": STR, "text": STR, "source_url": (STR, NONE)},
    "image_manifest.jsonl": {"image_id": STR, "doc_id": STR, "path": STR, "source_url": (STR, NONE),
                             "passage_ids": list, "width": (int, NONE), "height": (int, NONE), "ok": bool},
    "questions.jsonl": {"qid": STR, "question": STR, "answers": list, "gold_doc_ids": list,
                        "gold_image_ids": list, "gold_passage_ids": list, "distractor_image_ids": list,
                        "distractor_passage_ids": list, "q_type": (STR, NONE), "modalities": list,
                        "rephrasing_confidence": (int, float, NONE), "split": STR, "group_id": int},
}
LABEL_FIELDS = {"question", "qid", "answers", "split", "group_id", "gold_doc_ids", "gold_image_ids",
                "gold_passage_ids", "distractor_image_ids", "distractor_passage_ids"}
STANDALONE_BAD = re.compile(r"\b(this|these|the (following|given|above|below))\s+"
                            r"(image|images|picture|pictures|photo|photograph|figure)\b", re.I)


def load(path, errors):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                errors.append(f"{path.name}: line {n} is not valid JSON")
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--skip-image-open", action="store_true", help="only check that files exist")
    ap.add_argument("--show", type=int, default=0, help="print N random questions with their evidence")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.manifest_dir)
    errors, warns = [], []

    # 1. files, counts, hashes
    data = {}
    for name in SCHEMA:
        if not (d / name).exists():
            errors.append(f"missing file: {name}")
        else:
            data[name] = load(d / name, errors)
    if len(data) < 3:
        print("\n".join("ERROR " + e for e in errors)); sys.exit(1)
    texts, images, qs = (data["text_manifest.jsonl"], data["image_manifest.jsonl"], data["questions.jsonl"])

    rp = d / "report.json"
    if rp.exists():
        rep = json.loads(rp.read_text())
        for name, h in (rep.get("sha256") or {}).items():
            if sha256(d / name) != h:
                errors.append(f"hash mismatch for {name} (file changed since report.json was written)")
        for key, n in (("corpus_passages", len(texts)), ("corpus_images", len(images)),
                       ("n_questions", len(qs))):
            if key in rep and rep[key] != n:
                errors.append(f"report.json {key}={rep[key]} but file has {n}")
        sizes = rep.get("split_sizes", {})
        if sizes and sum(sizes.values()) != len(qs):
            errors.append(f"report split_sizes {sizes} do not add up to {len(qs)} questions")
    else:
        warns.append("report.json not found: hashes not verified")

    # 2. schema
    for name, rows in data.items():
        spec = SCHEMA[name]
        bad = Counter()
        for r in rows:
            for k, t in spec.items():
                if k not in r:
                    bad[f"missing key '{k}'"] += 1
                elif not isinstance(r[k], t) or (t is int and isinstance(r[k], bool)):
                    bad[f"key '{k}' has type {type(r[k]).__name__}"] += 1
            for k in r.keys() - spec.keys():
                bad[f"unexpected key '{k}'"] += 1
        for msg, n in bad.items():
            (warns if msg.startswith("unexpected") else errors).append(f"{name}: {msg} in {n} rows")

    # 3. uniqueness
    for name, key, rows in (("text_manifest", "passage_id", texts), ("image_manifest", "image_id", images),
                            ("questions", "qid", qs)):
        c = Counter(r.get(key) for r in rows)
        dup = [k for k, v in c.items() if v > 1]
        if dup:
            errors.append(f"{name}: {len(dup)} duplicate {key} values, e.g. {dup[:3]}")

    # 4. referential integrity
    pid, iid = {r["passage_id"] for r in texts}, {r["image_id"] for r in images}
    docs = {r["doc_id"] for r in texts} | {r["doc_id"] for r in images}
    for r in images:
        miss = [p for p in r.get("passage_ids", []) if p not in pid]
        if miss:
            errors.append(f"image {r['image_id']}: passage_ids not in text manifest: {miss[:3]}")
    for q in qs:
        for key, universe in (("gold_image_ids", iid), ("distractor_image_ids", iid),
                              ("gold_passage_ids", pid), ("distractor_passage_ids", pid),
                              ("gold_doc_ids", docs)):
            miss = [x for x in q.get(key, []) if x not in universe]
            if miss:
                errors.append(f"question {q['qid']}: {key} not in corpus: {miss[:3]}")
        if set(q.get("gold_image_ids", [])) & set(q.get("distractor_image_ids", [])):
            errors.append(f"question {q['qid']}: gold and distractor images overlap")
        if set(q.get("gold_passage_ids", [])) & set(q.get("distractor_passage_ids", [])):
            errors.append(f"question {q['qid']}: gold and distractor passages overlap")
        if not q.get("gold_image_ids"):
            errors.append(f"question {q['qid']}: no gold image")
        if not [a for a in q.get("answers", []) if str(a).strip()]:
            errors.append(f"question {q['qid']}: empty answers")
        if not str(q.get("question", "")).strip():
            errors.append(f"question {q['qid']}: empty question text")

    # 5. image files
    n_bad_img = 0
    if args.images_dir:
        base = Path(args.images_dir)
        for r in images:
            p = base / r["path"]
            if not p.exists():
                n_bad_img += 1
                continue
            if not args.skip_image_open:
                try:
                    from PIL import Image
                    with Image.open(p) as im:
                        w, h = im.size
                        im.verify()
                    if r.get("width") and (r["width"], r["height"]) != (w, h):
                        errors.append(f"image {r['image_id']}: size in manifest {r['width']}x{r['height']} != file {w}x{h}")
                except Exception:
                    n_bad_img += 1
        if n_bad_img:
            errors.append(f"{n_bad_img} images missing or unreadable under {base}")
    else:
        warns.append("--images-dir not given: image files not checked")
    flagged = sum(1 for r in images if not r.get("ok", True))
    if flagged:
        errors.append(f"{flagged} images flagged ok=false in image_manifest")

    # 6. split hygiene
    by_split = {s: [q for q in qs if q.get("split") == s] for s in {q.get("split") for q in qs}}
    unknown = set(by_split) - {"dev", "test"}
    if unknown:
        errors.append(f"unknown split names: {unknown}")
    dev_docs = {x for q in by_split.get("dev", []) for x in q["gold_doc_ids"]}
    test_docs = {x for q in by_split.get("test", []) for x in q["gold_doc_ids"]}
    if dev_docs & test_docs:
        errors.append(f"{len(dev_docs & test_docs)} gold documents appear in both dev and test (grouping violated)")
    grp = {}
    for q in qs:
        grp.setdefault(q["group_id"], set()).add(q["split"])
    if any(len(s) > 1 for s in grp.values()):
        errors.append("a group_id spans more than one split")

    # 7. leakage into index-side files
    for name, rows in (("text_manifest.jsonl", texts), ("image_manifest.jsonl", images)):
        leaked = set().union(*(r.keys() for r in rows)) & LABEL_FIELDS if rows else set()
        if leaked:
            errors.append(f"{name} contains label/question fields: {sorted(leaked)} (index must not see these)")

    # 8. warnings
    no_pass = sum(1 for r in images if not r.get("passage_ids"))
    if no_pass:
        warns.append(f"{no_pass}/{len(images)} images have no same-title passage (no image-passage edge)")
    bad_q = [q["qid"] for q in qs if STANDALONE_BAD.search(q.get("question", ""))]
    if bad_q:
        warns.append(f"{len(bad_q)} questions mention 'this image/picture/figure' (not standalone?) e.g. {bad_q[:3]}")
    empty_txt = sum(1 for r in texts if not r["text"].strip())
    if empty_txt:
        warns.append(f"{empty_txt} empty passages")

    # report
    print(f"passages {len(texts)} | images {len(images)} | docs {len(docs)} | questions {len(qs)} "
          f"({', '.join(f'{s}={len(v)}' for s, v in sorted(by_split.items()))})")
    if qs:
        di = [len(q['distractor_image_ids']) for q in qs]
        dp = [len(q['distractor_passage_ids']) for q in qs]
        print(f"distractor images/question: mean {st.mean(di):.1f} (min {min(di)}, max {max(di)}) | "
              f"distractor passages/question: mean {st.mean(dp):.1f}")
        print("q_type counts:", dict(Counter(q["q_type"] for q in qs).most_common(8)))
    if texts:
        L = sorted(len(r["text"].split()) for r in texts)
        print(f"passage words: median {L[len(L)//2]}, 95th pct {L[int(len(L)*.95)]}, max {L[-1]}")

    if args.show and qs:
        rng = random.Random(args.seed)
        ti = {r["passage_id"]: r for r in texts}
        ii = {r["image_id"]: r for r in images}
        for q in rng.sample(qs, min(args.show, len(qs))):
            print("\n" + "=" * 70)
            print(f"[{q['split']}] {q['qid']}  type={q['q_type']}  modalities={q['modalities']}")
            print("Q:", q["question"]); print("A:", q["answers"])
            for i in q["gold_image_ids"]:
                r = ii[i]
                print(f"  GOLD IMAGE {i}  doc='{r['doc_id']}'  file={r['path']}  linked passages={len(r['passage_ids'])}")
                for p in r["passage_ids"][:2]:
                    print("     passage:", ti[p]["text"][:140].replace("\n", " "))
            print(f"  gold passages: {len(q['gold_passage_ids'])} | distractor images: "
                  f"{len(q['distractor_image_ids'])} | distractor passages: {len(q['distractor_passage_ids'])}")

    print()
    for w in warns:
        print("WARN ", w)
    for e in errors[:50]:
        print("ERROR", e)
    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more errors")
    print(f"\n{'FAILED' if errors else 'PASSED'}: {len(errors)} errors, {len(warns)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
