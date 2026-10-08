#!/usr/bin/env python3
"""
Build the study manifests from the OFFICIAL MultiModalQA release (github.com/allenai/multimodalqa).

Inputs  (--data-dir, searched recursively; both naming schemes are accepted)
  MultiModalQA_train.jsonl.gz / MultiModalQA_dev.jsonl.gz   (or MMQA_*.jsonl.gz)
  texts.jsonl.gz, images.jsonl.gz                           (or MMQA_*.jsonl.gz)
  images (--images-dir): the unzipped final_dataset_images/ folder
The official test file has no labels, so only train + dev are used.

Outputs (--out-dir)
  text_manifest.jsonl      doc_id (= Wikipedia title), passage_id, text, source_url
  image_manifest.jsonl     image_id, doc_id, passage_ids (same-title paragraphs), path, size, ok
  questions.jsonl          qid, question, answers, gold ids, distractor ids, split, group_id, ...
  report.json              counts + sha256 of every manifest (freeze these hashes)

LEAKAGE GUARD: gold labels live ONLY in questions.jsonl. Index construction and captioning
must read only text_manifest.jsonl and image_manifest.jsonl.

Pipeline
  1. keep questions whose metadata.modalities contains "image" and not "table"
  2. map supporting_context ids -> Wikipedia titles (document = Wikipedia page title)
  3. group questions that share a gold document (union-find), so no group straddles dev/test
  4. shuffle groups with --seed, fill dev then test
  5. corpus = union of each selected question's full context (gold + distractors)
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


# ----------------------------------------------------------------------------- io
def find_file(data_dir, *patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, "**", pat), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"none of {patterns} found under {data_dir}")


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def answer_strings(answers):
    """README shows answers both as plain strings and as objects with an 'answer' key."""
    out = []
    for a in answers or []:
        s = a.get("answer", "") if isinstance(a, dict) else a
        s = str(s).strip()
        if s:
            out.append(s)
    return out



# ------------------------------------------------------------------ image files
def index_image_files(images_dir):
    """Index every file under images_dir so manifest paths can be resolved even when the
    case or extension differs (.JPG vs .jpg) or the files sit in a sub-folder."""
    root = Path(images_dir)
    by_rel, by_lower, by_stem = {}, {}, {}
    for p in sorted(root.rglob("*")):                 # sorted -> deterministic
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            by_rel[rel] = rel
            by_lower.setdefault(p.name.lower(), rel)
            by_stem.setdefault(p.stem.lower(), rel)
    return by_rel, by_lower, by_stem


def resolve_image(rec_path, idx):
    """Return the real relative path of an image, or None. Order: exact, case-insensitive, same stem."""
    by_rel, by_lower, by_stem = idx
    if rec_path in by_rel:
        return rec_path
    name = Path(rec_path).name
    return by_lower.get(name.lower()) or by_stem.get(Path(rec_path).stem.lower())

# ------------------------------------------------------------------------ grouping
def group_questions(qs):
    """Union-find: questions sharing any gold document end up in one group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for q in qs:
        k = ("q", q["qid"])
        find(k)
        for d in q["gold_doc_ids"]:
            union(k, ("d", d))
    groups = defaultdict(list)
    for q in qs:
        groups[find(("q", q["qid"]))].append(q)
    return list(groups.values())


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-dev", type=int, default=100)
    ap.add_argument("--num-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop questions whose rephrasing_meta.confidence is lower (default: keep all)")
    ap.add_argument("--skip-image-check", action="store_true", help="do not open image files (faster)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed}

    if not Path(args.images_dir).is_dir():
        raise SystemExit(f"--images-dir does not exist: {Path(args.images_dir).resolve()}")
    file_index = index_image_files(args.images_dir)
    print(f"{len(file_index[0])} files found under {Path(args.images_dir).resolve()}")
    if not file_index[0]:
        raise SystemExit("--images-dir contains no files (is the zip extracted?)")

    # ---- load context files
    texts = {r["id"]: r for r in read_jsonl(find_file(args.data_dir, "*texts.jsonl*"))}
    images = {r["id"]: r for r in read_jsonl(find_file(args.data_dir, "*images.jsonl*"))}
    report["n_texts_total"], report["n_images_total"] = len(texts), len(images)

    # ---- load labelled questions (train + dev; the official test split has no labels)
    raw, seen = [], set()
    for pat in ("*train.jsonl*", "*dev.jsonl*"):
        for r in read_jsonl(find_file(args.data_dir, pat)):
            if r["qid"] not in seen:
                seen.add(r["qid"])
                raw.append(r)
    report["n_questions_loaded"] = len(raw)

    # ---- filter + normalise
    qs, reasons = [], Counter()
    for r in raw:
        meta = r.get("metadata") or {}
        mods = set(meta.get("modalities") or [])
        if "image" not in mods:
            reasons["no_image_modality"] += 1
            continue
        if "table" in mods:
            reasons["needs_table"] += 1
            continue
        if "TableQ" in (meta.get("type") or ""):      # e.g. Compare(Compose(TableQ,ImageQ),...) needs a table step
            reasons["table_step_in_question_type"] += 1
            continue
        conf = (meta.get("rephrasing_meta") or {}).get("confidence")
        if conf is not None and conf < args.min_confidence:
            reasons["low_rephrasing_confidence"] += 1
            continue
        gold_images, gold_texts, miss = [], [], 0
        for sc in r.get("supporting_context") or []:
            part, did = sc.get("doc_part"), sc.get("doc_id")
            if part == "image" and did in images:
                gold_images.append(did)
            elif part == "text" and did in texts:
                gold_texts.append(did)
            elif part in ("image", "text"):
                miss += 1
        if not gold_images or miss:
            reasons["gold_missing_or_unmapped"] += 1
            continue
        titles = sorted({images[i]["title"].strip() for i in gold_images} |
                        {texts[t]["title"].strip() for t in gold_texts})
        ctx_img = set(meta.get("image_doc_ids") or []) | set(gold_images)
        ctx_txt = set(meta.get("text_doc_ids") or []) | set(gold_texts)
        ans = answer_strings(r.get("answers"))
        if not ans:
            reasons["no_answer"] += 1
            continue
        qs.append({
            "qid": r["qid"], "question": r["question"], "answers": ans,
            "gold_doc_ids": titles, "gold_image_ids": sorted(set(gold_images)),
            "gold_passage_ids": sorted(set(gold_texts)),
            "distractor_image_ids": sorted(ctx_img - set(gold_images)),
            "distractor_passage_ids": sorted(ctx_txt - set(gold_texts)),
            "q_type": meta.get("type"), "modalities": sorted(mods),
            "rephrasing_confidence": conf,
        })
    report["n_questions_after_filter"] = len(qs)
    report["filter_drop_reasons"] = dict(reasons)
    report["q_type_counts"] = dict(Counter(q["q_type"] for q in qs))

    # ---- grouped split
    rng = random.Random(args.seed)
    groups = group_questions(qs)
    groups.sort(key=lambda g: g[0]["qid"])      # deterministic order before shuffling
    rng.shuffle(groups)
    split_of, sizes = {}, {"dev": 0, "test": 0}
    targets = {"dev": args.num_dev, "test": args.num_test}
    for gi, g in enumerate(groups):
        for name in ("dev", "test"):                      # fill dev first, then test
            if sizes[name] < targets[name] and sizes[name] + len(g) <= targets[name] * 1.1:
                for q in g:
                    split_of[q["qid"]] = name
                    q["split"], q["group_id"] = name, gi
                sizes[name] += len(g)
                break
    selected = [q for q in qs if q["qid"] in split_of]
    report["split_sizes"] = sizes
    report["n_groups_total"] = len(groups)
    if sizes["dev"] < args.num_dev or sizes["test"] < args.num_test:
        report["warning"] = "requested split sizes not reached; relax filters or targets"

    # ---- corpus = union of the selected questions' contexts (gold + distractors)
    img_ids, txt_ids = set(), set()
    for q in selected:
        img_ids |= set(q["gold_image_ids"]) | set(q["distractor_image_ids"])
        txt_ids |= set(q["gold_passage_ids"]) | set(q["distractor_passage_ids"])
    img_ids &= images.keys()
    txt_ids &= texts.keys()

    text_rows = [{"doc_id": texts[t]["title"].strip(), "passage_id": t, "text": texts[t]["text"],
                  "source_url": texts[t].get("url")} for t in sorted(txt_ids)]
    by_title = defaultdict(list)
    for r in text_rows:
        by_title[r["doc_id"]].append(r["passage_id"])

    image_rows, bad, adjusted = [], [], 0
    for i in sorted(img_ids):
        rec = images[i]
        rel = resolve_image(rec["path"], file_index)
        if rel is not None and rel != rec["path"]:
            adjusted += 1
        p = Path(args.images_dir) / (rel or rec["path"])
        # relative path on purpose: absolute paths would change the hash between your laptop and Snellius
        row = {"image_id": i, "doc_id": rec["title"].strip(), "path": rel or rec["path"],
               "source_url": rec.get("url"), "passage_ids": by_title.get(rec["title"].strip(), []),
               "width": None, "height": None, "ok": p.exists()}
        if row["ok"] and not args.skip_image_check:
            try:
                from PIL import Image
                with Image.open(p) as im:
                    row["width"], row["height"] = im.size
                    im.verify()
            except Exception:
                row["ok"] = False
        if not row["ok"]:
            bad.append(i)
        image_rows.append(row)

    write_jsonl(out / "text_manifest.jsonl", text_rows)
    write_jsonl(out / "image_manifest.jsonl", image_rows)
    write_jsonl(out / "questions.jsonl", selected)

    bad_set = set(bad)
    report.update({
        "corpus_passages": len(text_rows), "corpus_images": len(image_rows),
        "corpus_docs": len({r["doc_id"] for r in text_rows} | {r["doc_id"] for r in image_rows}),
        "images_missing_or_unreadable": len(bad),
        "images_path_adjusted_case_or_extension": adjusted,
        "images_without_same_title_passage": sum(1 for r in image_rows if not r["passage_ids"]),
        "questions_with_unusable_gold_image": sum(1 for q in selected if set(q["gold_image_ids"]) & bad_set),
        "mean_distractor_images_per_question":
            round(sum(len(q["distractor_image_ids"]) for q in selected) / max(1, len(selected)), 2),
        "mean_distractor_passages_per_question":
            round(sum(len(q["distractor_passage_ids"]) for q in selected) / max(1, len(selected)), 2),
        "sha256": {n: sha256(out / n) for n in
                   ("text_manifest.jsonl", "image_manifest.jsonl", "questions.jsonl")},
    })
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
