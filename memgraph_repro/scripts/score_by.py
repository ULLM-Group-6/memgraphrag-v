#!/usr/bin/env python3
import collections, json, re, string, sys

def norm(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

qa_path, q_path, field = sys.argv[1:4]
qs = {q["_id"]: q for q in json.load(open(q_path, encoding="utf-8"))}
recs = json.load(open(qa_path, encoding="utf-8"))["results"]
by = collections.defaultdict(lambda: [0, 0])
miss = 0
for r in recs:
    q = qs.get(r.get("_id") or r.get("id"))
    if q is None:
        miss += 1
        continue
    key = str(q.get(field))
    by[key][1] += 1
    by[key][0] += norm(r["answer"]) in norm(r["predicted_answer"])
print(f"unmatched records: {miss}")
for k, (a, n) in sorted(by.items()):
    print(f"{field}={k:22s} n={n:4d}  acc={100*a/n:5.1f}%")
