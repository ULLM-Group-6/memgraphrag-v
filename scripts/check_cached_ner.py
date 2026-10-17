"""Read-only NER parser/cache check. No models, server calls, or cache writes.

Run on Snellius after applying memgraphrag-ner-json.patch.
"""
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


def load_parser(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "_extract_ner_from_response"]
    if len(nodes) != 1:
        raise ValueError("Expected exactly one NER parser function")
    # Compile only this pure helper, not the imports/model classes in the module.
    namespace = {"json": json, "re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_extract_ner_from_response"]


def check_examples(parse):
    examples = [
        ('["LeBron James", "March 2011", "Washington Wizards", "Miami Heat"]',
         ["LeBron James", "March 2011", "Washington Wizards", "Miami Heat"]),
        ('[]\nExplanation: The provided text does not contain any named entities.', []),
        ('["sharifov", "shengfeng", "2016 Summer Olympics"]',
         ["sharifov", "shengfeng", "2016 Summer Olympics"]),
        ('{"named_entities": ["Paris"]}', ["Paris"]),
        ('{"named_entities": []}', []),
        ('```json\n["Paris"]\n```', ["Paris"]),
        ('```json\n{"named_entities": ["Paris"]}\n```', ["Paris"]),
        ('Here are the entities:\n["Paris"]', ["Paris"]),
        (json.dumps({"named_entities": ['Object [A] {B} "quoted"']}),
         ['Object [A] {B} "quoted"']),
    ]
    invalid = [
        'No entities.', '{"entities": []}', '{"named_entities": null}',
        '{"named_entities": "Paris"}', '[1]', '[{"name": "Paris"}]',
        '{"named_entities": ["Paris"]', '["Paris",',
        '{"named_entities": [["Paris"]]}', None,
    ]
    for response, expected in examples:
        actual = parse(response)
        if actual != expected:
            raise AssertionError((response, actual, expected))
    for response in invalid:
        try:
            parse(response)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid response accepted: {response!r}")
    print(f"Parser checks passed: {len(examples) + len(invalid)} cases.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path)
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    args = ap.parse_args()
    parse = load_parser(args.repo / "code/src/information_extraction/openie_openai.py")
    check_examples(parse)
    if args.artifacts is None:
        return
    root = args.artifacts / "index"
    report = json.loads((root / "text/openie_failures.json").read_text(encoding="utf-8"))
    failed = [f for f in report["chunks"]
              if any(e.startswith("ner:") for e in f["errors"])]
    wanted = {f["chunk_id"] for f in failed}
    chunks = {}
    with (root / "chunks.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row["chunk_id"] in wanted:
                    chunks[row["chunk_id"]] = row["text"]
    template = runpy.run_path(str(args.repo / "code/src/prompts/templates/ner.py"))["prompt_template"]
    cache = root / "text/llm_cache" / f"{args.model.replace('/', '_')}_cache.sqlite"
    db = sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
    counts, reasons, examples = Counter(), Counter(), []
    try:
        for failure in failed:
            cid = failure["chunk_id"]
            if cid not in chunks:
                counts["chunk_text_missing"] += 1
                continue
            messages = [{"role": t["role"],
                         "content": Template(t["content"]).substitute(passage=chunks[cid])}
                        for t in template]
            found = None
            for seed in (None, 0, 42):
                for temperature in (0, 0.0):
                    data = dict(messages=messages, model=args.model, seed=seed, temperature=temperature)
                    key = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
                    found = db.execute("SELECT message, metadata FROM cache WHERE key=?", (key,)).fetchone()
                    if found:
                        break
                if found:
                    break
            if found is None:
                counts["cache_key_not_found"] += 1
                continue
            counts["cached_responses_found"] += 1
            raw, meta = found[0], json.loads(found[1])
            reason = str(meta.get("finish_reason"))
            reasons[reason] += 1
            if reason != "stop":
                # Upstream has separate truncated-output repair; do not claim
                # that accepting this raw response validates that repair.
                counts["non_stop_requires_review"] += 1
                error = f"finish_reason={reason}"
            else:
                try:
                    parse(raw)
                    counts["stop_responses_parse_ok"] += 1
                    continue
                except ValueError as exc:
                    counts["parse_failures"] += 1
                    error = str(exc)
            if len(examples) < 3:
                examples.append({"chunk_id": cid, "error": error, "response": str(raw)[:2000]})
    finally:
        db.close()
    print(json.dumps({"failed_chunks_in_report": report["failed"],
                      "chunks_with_ner_errors": len(failed), "counts": dict(counts),
                      "finish_reasons": dict(reasons), "unresolved_examples": examples,
                      "note": "Read-only check. Triple extraction and remaining pipeline stages are not validated."},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
