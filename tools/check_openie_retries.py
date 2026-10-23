"""CPU-only checks of the applied OpenIE patch; no model or cache access."""
import __future__
import argparse
import ast
import json
import logging
import re
from types import SimpleNamespace
from pathlib import Path

from check_cached_ner import check_examples


def load_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_extract_ner_from_response", "_extract_triples_from_response", "_infer_openie_with_validation",
             "_parse_triples", "_clean_triple", "_decode_first_json", "_to_strict_json"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise ValueError("The OpenIE retry and triple-repair patches are not both applied")
    # Module-level constants the helpers use (_JSON_ESCAPES, _LITERAL_TAG, ...).
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and all(
        isinstance(t, ast.Name) and t.id.startswith("_") for t in n.targets)] + nodes
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenIE")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("ner", "triple_extraction")]
    nodes.append(cls)
    logger = logging.getLogger("offline-openie-check")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    namespace = dict(json=json, re=re, logger=logger,
                     NerRawOutput=SimpleNamespace, TripleRawOutput=SimpleNamespace,
                     filter_invalid_triples=lambda triples: [list(t) for t in dict.fromkeys(map(tuple, triples))])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def infer(self, *, messages):
        self.requests.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        raw, reason = response if isinstance(response, tuple) else (response, "stop")
        return raw, dict(finish_reason=reason, prompt_tokens=10, completion_tokens=5), len(self.requests) == 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    ns = load_functions(args.repo / "code/src/information_extraction/openie_openai.py")
    check_examples(ns["_extract_ner_from_response"])
    parse = ns["_extract_triples_from_response"]
    valid = [
        ('[]', []), ('{"triples": []}', []),
        ('[["A", "is", "B"]]', [["A", "is", "B"]]),
        ('```json\n{"triples": [["A", "is", "B"]]}\n```', [["A", "is", "B"]]),
        ('{"triples": [["A", "scored", "35.6"]]}', [["A", "scored", "35.6"]]),
        # Repaired: numbers, qualifiers beyond the object, near-JSON from Qwen.
        ('{"triples": [["A", "scored", 35.6]]}', [["A", "scored", "35.6"]]),
        ('{"triples": [["A", "played for", "B", "in", "2008"]]}', [["A", "played for", "B in 2008"]]),
        ('{"triples": [["A", "is", "B"] // assumption\n]}', [["A", "is", "B"]]),
        ('{"triples": [["A", "is", "B"],]}', [["A", "is", "B"]]),
        ('{"triples": [["A", "held on", "24 August 2021"^^xsd:date]]}', [["A", "held on", "24 August 2021"]]),
        ("{\"triples\": [[\"A\", \"sang\", '\"a line\"']]}", [["A", "sang", '"a line"']]),
        ('{"triples": [["\\u043b\\u441", "is", "B"]]}', [["\u043b\u0441", "is", "B"]]),
        # Unusable triples are dropped while the others are kept.
        ('{"triples": [["A", "is", null], ["C", "is", "D"]]}', [["C", "is", "D"]]),
        ('{"triples": [["A", "is", None], ["", "is", "D"], ["E", "is", "F"]]}', [["E", "is", "F"]]),
    ]
    for raw, expected in valid:
        assert parse(raw) == expected, (raw, parse(raw))
    # Not JSON at all, or every listed triple is unusable: retried, then failed.
    invalid = [
        'No triples', '{"players": []}', '{"triples": null}',
        '{"triples": [["A", "is", null]]}', '{"triples": [["A", "is", None]]}',
        '{"triples": [["A", "is", true]]}', '{"triples": [["A", "is", {"x": 1}]]}',
        '{"triples": [["A", "is", ["B"]]]}', '{"triples": [["A", "is"]]}',
        '{"triples": [["A", "is", ""]]}', '{"triples": [["A", "is", "   "]]}',
        '{"triples": ["ABC"]}',
        '{"triples": [["A"\uff0c "is", "B"]]}', '{"triples": [["A", "is", "B"]',
    ]
    for raw in invalid:
        try:
            parse(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid triples accepted: {raw}")
    infer = ns["_infer_openie_with_validation"]
    messages = [dict(role="user", content="Original passage")]
    fake = FakeLLM(['malformed', '{"triples": [["A", "is", "B"]]}'])
    raw, meta, parsed = infer(fake, messages, parse, "triples")
    assert parsed == [["A", "is", "B"]] and "error" not in meta
    assert len(fake.requests) == 2 and fake.requests[0] == messages
    assert fake.requests[1][:-1] == messages and len(messages) == 1
    assert len(meta["extraction_attempts"]) == 2
    assert "error" in meta["extraction_attempts"][0]
    fake = FakeLLM(['bad'] * 3)
    raw, meta, parsed = infer(fake, messages, parse, "triples")
    assert parsed is None and "error" in meta and len(fake.requests) == 3
    assert len({json.dumps(r) for r in fake.requests}) == 3
    fake = FakeLLM([('[]', 'length')] * 3)
    assert infer(fake, messages, parse, "triples")[2] is None
    fake = FakeLLM([RuntimeError("Server unavailable")])
    assert "Server unavailable" in infer(fake, messages, parse, "triples")[1]["error"]
    assert len(fake.requests) == 1
    fake = FakeLLM(['[]'])
    assert infer(fake, messages, parse, "triples")[2] == [] and len(fake.requests) == 1
    # Exercise the actual patched NER/triple methods with model-free stand-ins.
    backend = ns["OpenIE"]()
    backend.prompt_template_manager = SimpleNamespace(render=lambda **kwargs: messages)
    backend.llm_model = FakeLLM(['{"players": []}', '["A", "A"]'])
    result = backend.ner("chunk", "Original passage")
    assert result.unique_entities == ["A"] and "error" not in result.metadata
    backend.llm_model = FakeLLM(['bad'] * 3)
    result = backend.ner("chunk", "Original passage")
    assert result.unique_entities == [] and "error" in result.metadata
    backend.llm_model = FakeLLM(['{"triples": [["A", "is", null]]}', '[["A", "is", "B"]]'])
    result = backend.triple_extraction("chunk", "Original passage", ["A"])
    assert result.triples == [["A", "is", "B"]] and "error" not in result.metadata
    backend.llm_model = FakeLLM(['bad'] * 3)
    result = backend.triple_extraction("chunk", "Original passage", ["A"])
    assert result.triples == [] and "error" in result.metadata
    print(f"Passed {len(valid) + len(invalid)} triple parser cases and 9 retry/integration scenarios.")
    print("No models loaded, server calls made, or extraction caches changed.")


if __name__ == "__main__":
    main()
