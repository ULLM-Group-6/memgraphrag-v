#!/usr/bin/env python3
import json, math, re, string, sys

def norm(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def has(g, text):
    return (' ' + g + ' ') in (' ' + text + ' ')

def final_part(ans):
    parts = re.split(r"(?i)answer\s*:", ans)
    return parts[-1] if len(parts) > 1 else ans

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    d = json.load(open(path, encoding="utf-8"))
    recs = d.get("results") or []
    if not recs or "predicted_answer" not in recs[0] or "answer" not in recs[0]:
        print(f"{name}: unexpected structure; keys={list(d)}; first result keys={list(recs[0]) if recs else None}")
        continue
    full = fin = n = 0
    for r in recs:
        gold = r["answer"]
        golds = gold if isinstance(gold, list) else [gold]
        golds = [norm(g) for g in golds if str(g).strip()]
        pred = str(r["predicted_answer"] or "")
        if not golds: continue
        n += 1
        full += any(has(g, norm(pred)) for g in golds)
        fin += any(has(g, norm(final_part(pred))) for g in golds)
    lo1, hi1 = wilson(full, n); lo2, hi2 = wilson(fin, n)
    print(f"{name:16s} n={n:5d}  full-output {100*full/n:5.1f}% [{100*lo1:.1f}-{100*hi1:.1f}]   final-answer {100*fin/n:5.1f}% [{100*lo2:.1f}-{100*hi2:.1f}]")
