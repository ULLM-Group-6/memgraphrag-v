"""Inspect cached triple failures without loading models or changing caches."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy
import sqlite3
from string import Template

from check_cached_ner import load_parser
from check_openie_retries import load_functions


def lookup(db, messages, model):
    for seed in (None, 0, 42):
        for temperature in (0, 0.0):
            data = dict(messages=messages, model=model, seed=seed, temperature=temperature)
            key = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
            row = db.execute("SELECT message, metadata FROM cache WHERE key=?", (key,)).fetchone()
            if row is not None:
                return row[0], json.loads(row[1])
    return None


def classify(raw):
    if not isinstance(raw, str):
        return "not_text"
    start = re.search(r"[\[{]", raw)
    if start is None:
        return "no_json"
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start.start():])
    except json.JSONDecodeError as exc:
        return f"invalid_json: {exc.msg}"
    if isinstance(value, dict):
        if "triples" not in value:
            return "missing_triples_field"
        value = value["triples"]
    if not isinstance(value, list):
        return "triples_not_a_list"
    if any(not isinstance(t, list) or len(t) != 3 for t in value):
        return "invalid_triple_shape"
    if any(not isinstance(item, str) for t in value for item in t):
        return "non_string_triple_values"
    return "valid_json_and_triple_shape"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    args = ap.parse_args()
    root = args.artifacts / "index"
    report = json.loads((root / "text/openie_failures.json").read_text(encoding="utf-8"))
    failures = [c for c in report["chunks"]
                if any(e.startswith("triples:") for e in c["errors"])
                and not any(e.startswith("ner:") for e in c["errors"])]
    ids = {c["chunk_id"] for c in failures}
    chunks = {}
    with (root / "chunks.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row["chunk_id"] in ids:
                    chunks[row["chunk_id"]] = row["text"]
    src = args.repo / "code/src"
    parse_ner = load_parser(src / "information_extraction/openie_openai.py")
    ner = runpy.run_path(str(src / "prompts/templates/ner.py"))
    # Read literal prompt strings only: no upstream/model imports.
    tree = ast.parse((src / "prompts/templates/triple_extraction.py").read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = ast.literal_eval(node.value)
    frame = constants["ner_conditioned_re_frame"]
    prefix = [
        dict(role="system", content=constants["ner_conditioned_re_system"]),
        dict(role="user", content=frame.format(passage=ner["one_shot_ner_paragraph"],
                                               named_entity_json=ner["one_shot_ner_output"])),
        dict(role="assistant", content=constants["ner_conditioned_re_output"]),
    ]
    # With memgraphrag-openie-triple-repair.patch applied, also report what its
    # parser makes of each cached response (lookups need the unpatched prompt).
    try:
        repair = load_functions(src / "information_extraction/openie_openai.py")["_parse_triples"]
    except ValueError:
        repair = None
    repaired = Counter()
    cache = root / "text/llm_cache" / f"{args.model.replace('/', '_')}_cache.sqlite"
    db = sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
    counts, examples, seen = Counter(), [], Counter()
    try:
        for failure in failures:
            cid = failure["chunk_id"]
            raw, meta = None, None
            try:
                passage = chunks[cid]
                messages = [dict(role=t["role"], content=Template(t["content"]).substitute(passage=passage))
                            for t in ner["prompt_template"]]
                cached = lookup(db, messages, args.model)
                if cached is None:
                    raise ValueError("NER cache key not found")
                if cached[1].get("finish_reason") != "stop":
                    raise ValueError("NER finish reason needs separate review")
                entities = list(dict.fromkeys(parse_ner(cached[0])))
                messages = prefix + [dict(role="user", content=frame.format(
                    passage=passage, named_entity_json=json.dumps({"named_entities": entities})))]
                cached = lookup(db, messages, args.model)
                if cached is None:
                    raise ValueError("Triple cache key not found")
                raw, meta = cached
                status = classify(raw) if meta.get("finish_reason") == "stop" else "non_stop_requires_review"
            except (ValueError, KeyError) as exc:
                status = f"lookup_problem: {exc}"
            counts[status] += 1
            outcome = None
            if repair is not None and isinstance(raw, str) and meta.get("finish_reason") == "stop":
                try:
                    kept, dropped = repair(raw)
                    outcome = "kept_some" if kept else ("all_invalid_retry" if dropped else "empty_list")
                    repaired["dropped_triples"] += len(dropped)
                    repaired["kept_triples"] += len(kept)
                except ValueError as exc:
                    outcome = f"still_fails_retry: {exc}"
                repaired[outcome] += 1
            # Two examples of each outcome, bounded raw text for easy sharing.
            if seen[status] < 2 or (outcome and not outcome.startswith("kept") and seen[outcome] < 3):
                examples.append(dict(chunk_id=cid, original_errors=failure["errors"],
                                     assessment=status, repair_outcome=outcome, metadata=meta,
                                     response=raw[:6000] if isinstance(raw, str) else raw))
                seen[status] += 1
                seen[outcome] += 1
    finally:
        db.close()
    print(json.dumps(dict(chunks_checked=len(failures), counts=dict(counts),
                          with_triple_repair_patch=dict(repaired) if repair else "patch not applied",
                          examples=examples,
                          note="Read-only diagnostic; no cache changes applied."),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
