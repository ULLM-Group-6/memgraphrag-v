"""CPU-only checks of the memory-JSON patch; no model, server or cache access."""
import argparse
import ast
import json
from pathlib import Path
import re

from check_openie_retries import load_functions


def load_memory_json(repo):
    src = repo / "code/src"
    tree = ast.parse((src / "MemGraphRAG.py").read_text(encoding="utf-8"))
    error = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MemoryJSONError"]
    if len(error) != 1:
        raise ValueError("The memory-JSON patch is not applied: MemoryJSONError is missing")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MemGraphRAG")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_memory_json")
    method.decorator_list = []
    openie = load_functions(src / "information_extraction/openie_openai.py")
    namespace = dict(json=json, re=re, Dict=dict, Any=object,
                     _decode_first_json=openie["_decode_first_json"],
                     # Upstream's repair only closes brackets; identity is enough here.
                     fix_broken_generated_json=lambda text: text)
    exec(compile(ast.Module(body=error + [method], type_ignores=[]), "MemGraphRAG.py", "exec"), namespace)
    return namespace["_memory_json"], namespace["MemoryJSONError"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    parse, error = load_memory_json(args.repo)
    triple = {"ontology_triples": [{"triple": ["A", "is", "B"], "ontology": ["Person", "is", "Thing"]}]}
    valid = [
        json.dumps(triple),
        "```json\n" + json.dumps(triple) + "\n```",
        "{\n  \"ontology_triples\": [\n    {\n      'triple': ['A', 'is', 'B'],\n      'ontology': ['Person', 'is', 'Thing']\n    }\n  ]\n}",
        "{\n  \"ontology_triples\": [\n    {\n      // the schema\n      \"triple\": [\"A\", \"is\", \"B\"],\n"
        "      \"ontology\": [\"Person\", \"is\", \"Thing\"],\n    },\n  ]\n}",
        "Here is the result:\n" + json.dumps(triple) + "\nI hope it's useful.",
    ]
    for raw in valid:
        assert parse(raw) == triple, (raw, parse(raw))
    assert parse('{"conflicting_triple_ids": [1, 2], "conflicts": [], "note": None}')["note"] is None
    for raw in ("No JSON here", "[1, 2, 3]", '{"ontology_triples": [{"triple": ["A" "B"]'):
        try:
            parse(raw)
        except error:
            pass
        else:
            raise AssertionError(f"Accepted: {raw}")
    assert issubclass(error, ValueError)
    print(f"Passed {len(valid) + 4} memory-JSON cases. No models loaded or caches changed.")


if __name__ == "__main__":
    main()
