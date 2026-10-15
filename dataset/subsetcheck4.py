#!/usr/bin/env python3
"""
Select the WebQA graph-building set (default 600 questions) and a test set (default 200 questions).

  python select_webqa_subset.py --json WebQA_train_val.json --out-dir webqa_subset \
        [--n-build 600] [--n-test 200] [--mix 0.6,0.2,0.2] [--ner regex|spacy] [--seed 42]

DEFINITIONS
  source          one image source (caption + title + imgUrl) or one text snippet
  evidence class  by GOLD evidence: image (positive sources are images only), both (images and text), text (text only)
  corpus          union of ALL sources (positive + negative) of the graph-building questions
  entity          normalised named-entity string extracted from a source's caption / snippet (never from questions or answers)
  genuine entity  occurs in 2..--max-df-global sources of the whole file, after dropping months, weekdays, sentence-starters
                  and (with spaCy) dates, numbers, money, quantities

WHAT IS GUARANTEED (all asserted before anything is written)
  * every positive source of every test question is already inside the corpus built from the graph-building questions
  * no test question is a graph-building question (by qid AND by normalised question text)
  * class quotas are applied to BOTH sets; if a class cannot be filled the shortfall is reported, quotas are NOT weakened
    (unless you pass --fill-shortfall, which is then flagged as weakened in report.json)
  * the corpus manifests contain no question, answer, gold flag or distractor flag

HOW CONNECTIVITY IS MAXIMISED (heuristic greedy, not an optimum)
  1. test questions are chosen first, preferring ones whose positive sources are cheap to cover; for each, the graph-building
     questions that cover its positive sources ("coverers") are chosen so that they also attach to the corpus built so far
  2. the rest of the graph-building set is grown from the frontier: at each step add the question with the highest attachment
     = shared sources + shared entities + cross-modal links, with a penalty for newly added images that do not attach to the current corpus
  3. a random selection with the same quotas is evaluated for comparison

OUTPUTS in --out-dir
  graph_corpus/text_manifest.jsonl     caption passages (cap_<image_id>) and text snippets (txt_<snippet_id>)
  graph_corpus/image_manifest.jsonl    image_id, doc_id, passage_ids, source_url (imgUrl), commons_url   (no labels)
  graph_corpus/source_ids.json         all image ids and text-snippet ids in the corpus
  graph_corpus/images_to_download.jsonl   the ONLY images you need to download
  build_qa.json                        graph-building questions with answers, gold ids, gold documents, distractor ids
  test_qa.json                         test questions with answers, gold ids, gold documents (kept apart from the corpus)
  report.json                          counts, shortfalls, connectivity vs random baseline, source-overlap report
"""
import argparse
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("image", "both", "text")
MONTHS = "january february march april may june july august september october november december".split()
DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
STOPWORDS = set(MONTHS + DAYS + (
    "the a an this that these those there here it he she they we you i his her its their our one two first second new old "
    "in on at of for from with by as and or but not no yes view photo image file caption left right front back early late "
    "day night north south east west upper lower during after before when while where which who what").split())
CONNECTORS = {"of", "the", "de", "la", "von", "van", "and", "del"}
DROP_LABELS = {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}


# --------------------------------------------------------------------------- text helpers
def unquote(s):
    s = str(s).strip()
    return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s


def norm_text(s):
    return re.sub(r"\W+", " ", unquote(s).lower()).strip()


def clean_caption(s):
    s = re.sub(r"\.(jpe?g|png|tiff?|gif|svg)\b", "", str(s), flags=re.I)
    s = re.sub(r"(?<=\w)[-_](?=\w)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _cap_follows(toks, j):
    while j < len(toks) and toks[j].strip(".,;:()[]\"'").lower() in CONNECTORS:
        j += 1
    return j < len(toks) and toks[j].strip(".,;:()[]\"'")[:1].isupper()


def regex_entities(text):
    out, run, toks = [], [], text.split()
    for i, t in enumerate(toks):
        w = t.strip(".,;:()[]\"'")
        if w[:1].isupper():
            run.append(w)
        elif w.lower() in CONNECTORS and run and _cap_follows(toks, i + 1):
            run.append(w)
        else:
            if run:
                out.append(" ".join(run)); run = []
    if run:
        out.append(" ".join(run))
    res = []
    for e in out:
        n = re.sub(r"^(the|a|an) ", "", e.lower()).strip()
        if len(n) < 3 or n in STOPWORDS or n.isdigit():
            continue
        res.append(n)
    return res


def extract_entities(texts, mode, spacy_model, n_process):
    if mode == "spacy":
        import spacy
        nlp = spacy.load(spacy_model, disable=["parser", "lemmatizer"])
        out = []
        for doc in nlp.pipe(texts, batch_size=256, n_process=n_process):
            ents = set()
            for e in doc.ents:
                if e.label_ in DROP_LABELS:
                    continue
                n = re.sub(r"^(the|a|an) ", "", e.text.lower()).strip()
                if len(n) >= 3 and n not in STOPWORDS and not n.isdigit():
                    ents.add(n)
            out.append(ents)
        return out
    return [set(regex_entities(t)) for t in texts]


# ------------------------------------------------------------------------------- loading
def load(args):
    raw = json.load(open(args.json, encoding="utf-8"))
    items = list(raw.items()) if isinstance(raw, dict) else [(r.get("Guid"), r) for r in raw]
    sources, Q, skipped = {}, [], Counter()

    def add_img(f):
        sid = f"img:{f['image_id']}"
        sources.setdefault(sid, {"type": "img", "title": f.get("title", ""), "caption": f.get("caption", ""),
                                 "url": f.get("url"), "imgUrl": f.get("imgUrl"), "image_id": str(f["image_id"])})
        return sid

    def add_txt(f):
        text = next((str(f[k]) for k in ("fact", "snippet", "text", "content") if f.get(k)), "")
        if f.get("snippet_id") is None or not text:
            skipped["text fact without id or text"] += 1
            return None
        sid = f"txt:{f['snippet_id']}"
        sources.setdefault(sid, {"type": "txt", "title": f.get("title", ""), "text": text, "url": f.get("url"),
                                 "snippet_id": str(f["snippet_id"])})
        return sid

    for key, r in items:
        ans = [unquote(a) for a in r.get("A", []) if str(a).strip()]
        pos = [add_img(f) for f in r.get("img_posFacts", [])] + [x for x in (add_txt(f) for f in r.get("txt_posFacts", [])) if x]
        neg = [add_img(f) for f in r.get("img_negFacts", [])] + [x for x in (add_txt(f) for f in r.get("txt_negFacts", [])) if x]
        if not ans or not pos:
            skipped["no answer or no positive source"] += 1
            continue
        has_i = any(s.startswith("img:") for s in pos)
        has_t = any(s.startswith("txt:") for s in pos)
        cls = "both" if has_i and has_t else ("image" if has_i else "text")
        allsrc = list(dict.fromkeys(pos + neg))
        Q.append({"i": len(Q), "qid": str(r.get("Guid", key)), "question": unquote(r.get("Q", "")), "answers": ans,
                  "qcate": r.get("Qcate"), "topic": r.get("topic"), "split": r.get("split"), "cls": cls,
                  "pos": list(dict.fromkeys(pos)), "neg": [s for s in dict.fromkeys(neg) if s not in pos],
                  "S": allsrc, "Sset": frozenset(allsrc), "qnorm": norm_text(r.get("Q", ""))})
    return sources, Q, skipped


def source_text(s):
    return clean_caption(s.get("caption") or s.get("title") or "") if s["type"] == "img" else s["text"]


# ----------------------------------------------------------------------------- selection
def quotas(n, mix):
    q = {c: int(n * m) for c, m in zip(CLASSES, mix)}
    q["image"] += n - sum(q.values())          # rounding remainder goes to the image class
    return q


def select(args, Q, ents, q_img, q_txt, q_all, src_to_q, pos_of, ent_to_q, coverable, qB, qT, seed):
    rng = random.Random(seed)
    B, T, inB, inT = [], [], set(), set()
    cB, cT = Counter(), Counter()
    corpus, C_img, C_txt, C_all = set(), set(), set(), set()
    frontier, touched, dead = set(), set(), set()
    stats = Counter()

    def add_to_corpus(i):
        for s in Q[i]["S"]:
            if s not in corpus:
                corpus.add(s)
                touched.update(pos_of.get(s, ()))
                frontier.update(src_to_q[s])
        new = q_all[i] - C_all
        for e in new:
            frontier.update(ent_to_q[e])
        C_img.update(q_img[i]); C_txt.update(q_txt[i]); C_all.update(q_all[i])

    def attach(i):
        a = len(q_all[i] & C_all)
        cross = len(q_img[i] & C_txt) + len(q_txt[i] & C_img)
        shared = sum(1 for s in Q[i]["S"] if s in corpus)
        # Penalize new image sources that have no entity overlap with the
        # current corpus. This steers selection toward connected images rather
        # than merely accumulating sources. It is a heuristic, not a guarantee.
        new_imgs = [s for s in Q[i]["S"] if s.startswith("img:") and s not in corpus]
        unattached_imgs = sum(1 for s in new_imgs if not (ents[s] & C_all))
        # Reward candidates that connect their image entities to existing text.
        image_to_text = sum(1 for s in new_imgs if ents[s] & C_txt)
        return (a + args.w_cross * cross + args.w_source * shared
                + args.w_image_text * image_to_text
                - args.w_unattached_image * unattached_imgs)

    def add_B(i):
        B.append(i); inB.add(i); cB[Q[i]["cls"]] += 1; add_to_corpus(i)

    # ---- phase 1: test questions first, with the coverers that make their evidence present
    cand = {c: [i for i in coverable if Q[i]["cls"] == c] for c in CLASSES}
    while sum(cT.values()) < sum(qT.values()):
        open_cls = [c for c in CLASSES if cT[c] < qT[c]]
        ok = lambda j: (Q[j]["cls"] in open_cls and j in coverable_set and j not in inB and j not in inT and j not in dead)
        pool = {j for j in touched if ok(j)}
        for c in open_cls:
            pool.update(rng.sample(cand[c], min(args.cand_sample, len(cand[c]))))
        pool = [j for j in pool if ok(j)]
        if not pool:
            break
        j = min(pool, key=lambda x: (sum(1 for s in Q[x]["pos"] if s not in corpus), -attach(x), rng.random()))
        unc = [s for s in Q[j]["pos"] if s not in corpus]
        chosen, covered, tmp = [], set(), Counter()
        feasible = True
        for s in unc:
            if s in covered:
                continue
            opts = [b for b in src_to_q[s] if b != j and b not in inT and b not in inB and b not in chosen
                    and cB[Q[b]["cls"]] + tmp[Q[b]["cls"]] < qB[Q[b]["cls"]]]
            if not opts:
                feasible = False
                break
            left = set(unc) - covered
            b = max(opts, key=lambda x: (len(Q[x]["Sset"] & left) * 10 + attach(x) + rng.random()))
            chosen.append(b); covered |= Q[b]["Sset"]; tmp[Q[b]["cls"]] += 1
        if not feasible:
            dead.add(j)
            stats["test candidates dropped (no admissible coverer)"] += 1
            continue
        T.append(j); inT.add(j); cT[Q[j]["cls"]] += 1
        for b in chosen:
            add_B(b)
        stats["coverers added"] += len(chosen)

    # ---- phase 2: grow the graph-building set by attachment
    while sum(cB.values()) < sum(qB.values()):
        open_cls = [c for c in CLASSES if cB[c] < qB[c]]
        cands = [i for i in frontier if Q[i]["cls"] in open_cls and i not in inB and i not in inT]
        if not cands:
            pool = [i for c in open_cls for i in all_by_cls[c] if i not in inB and i not in inT]
            if not pool:
                break
            add_B(rng.choice(pool))
            stats["cold-start picks (frontier empty)"] += 1
            continue
        if len(cands) > args.frontier_sample:
            cands = rng.sample(cands, args.frontier_sample)
        add_B(max(cands, key=lambda x: (attach(x), rng.random())))
    return B, T, corpus, stats


def connectivity(corpus, ents, max_df, gold_img):
    df = Counter(e for s in corpus for e in ents[s])
    bridges = {e for e, c in df.items() if 2 <= c <= max_df}
    parent = {s: s for s in corpus}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first, ent_has_txt = {}, defaultdict(bool)
    for s in corpus:
        for e in ents[s] & bridges:
            if s.startswith("txt:"):
                ent_has_txt[e] = True
            if e in first:
                ra, rb = find(s), find(first[e])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[e] = s
    comp = Counter(find(s) for s in corpus)
    giant = comp.most_common(1)[0][0] if comp else None
    imgs = [s for s in corpus if s.startswith("img:")]
    txts = [s for s in corpus if s.startswith("txt:")]
    pct = lambda a, b: round(a / b, 3) if b else None
    gi = [s for s in gold_img if s in corpus]
    return {
        "sources": len(corpus), "images": len(imgs), "text_snippets": len(txts), "bridging_entities": len(bridges),
        "giant_component_share_of_sources": pct(comp[giant] if giant else 0, len(corpus)),
        "images_with_a_shared_entity": pct(sum(1 for s in imgs if ents[s] & bridges), len(imgs)),
        "images_in_giant_component": pct(sum(1 for s in imgs if find(s) == giant), len(imgs)),
        "images_sharing_entity_with_a_text_snippet": pct(sum(1 for s in imgs if any(ent_has_txt[e] for e in ents[s] & bridges)), len(imgs)),
        "text_snippets_in_giant_component": pct(sum(1 for s in txts if find(s) == giant), len(txts)),
        "gold_images_in_giant_component": pct(sum(1 for s in gi if find(s) == giant), len(gi)),
        "gold_images_sharing_entity_with_a_text_snippet": pct(
            sum(
                1 for s in gi
                if any(ent_has_txt[e] for e in ents[s] & bridges)
            ),
            len(gi)
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-build", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--mix", default="0.6,0.2,0.2", help="image-only, both, text-only fractions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ner", choices=["regex", "spacy"], default="regex")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--n-process", type=int, default=4)
    ap.add_argument("--max-df-global", type=int, default=200, help="entities in more sources than this are hubs")
    ap.add_argument("--max-df-corpus", type=int, default=50)
    ap.add_argument("--w-source", type=float, default=3.0)
    ap.add_argument("--w-cross", type=float, default=2.0)
    ap.add_argument("--w-image-text", type=float, default=5.0, help="bonus per new image sharing entities with existing text")
    ap.add_argument("--w-unattached-image", type=float, default=5.0, help="penalty per new image with no entity overlap with current corpus")
    ap.add_argument("--target-image-connectivity", type=float, default=0.80, help="target fraction of corpus images in the giant component; reported, not guaranteed")
    ap.add_argument("--cand-sample", type=int, default=300)
    ap.add_argument("--frontier-sample", type=int, default=1500)
    ap.add_argument("--fill-shortfall", choices=["none", "image"], default="none",
                    help="'image' moves unfilled quota into the image class; this WEAKENS the criteria and is flagged in the report")
    args = ap.parse_args()
    mix = [float(x) for x in args.mix.split(",")]
    assert abs(sum(mix) - 1) < 1e-6 and len(mix) == 3

    sources, Q, skipped = load(args)
    print(f"questions with answer and positive evidence: {len(Q)} | sources: {len(sources)} | skipped: {dict(skipped)}")
    cls_count = Counter(q["cls"] for q in Q)
    print("evidence classes in the file:", {c: cls_count[c] for c in CLASSES})

    # ---- entities
    sids = list(sources)
    raw_ents = extract_entities([source_text(sources[s]) for s in sids], args.ner, args.spacy_model, args.n_process)
    df = Counter(e for es in raw_ents for e in es)
    genuine = {e for e, c in df.items() if 2 <= c <= args.max_df_global}
    ents = {s: es & genuine for s, es in zip(sids, raw_ents)}
    print(f"NER: {args.ner} | genuine entities: {len(genuine)} (of {len(df)} distinct)")

    q_img, q_txt, q_all = {}, {}, {}
    ent_to_q, src_to_q, pos_of = defaultdict(list), defaultdict(list), defaultdict(list)
    for q in Q:
        i = q["i"]
        q_img[i] = set().union(*(ents[s] for s in q["S"] if s.startswith("img:"))) if q["S"] else set()
        q_txt[i] = set().union(*(ents[s] for s in q["S"] if s.startswith("txt:"))) if q["S"] else set()
        q_all[i] = q_img[i] | q_txt[i]
        for e in q_all[i]:
            ent_to_q[e].append(i)
        for s in q["S"]:
            src_to_q[s].append(i)
        for s in q["pos"]:
            pos_of[s].append(i)

    global coverable_set, all_by_cls
    coverable = [q["i"] for q in Q if all(any(j != q["i"] for j in src_to_q[s]) for s in q["pos"])]
    coverable_set = set(coverable)
    all_by_cls = {c: [q["i"] for q in Q if q["cls"] == c] for c in CLASSES}
    cov_count = Counter(Q[i]["cls"] for i in coverable)
    print("test-eligible (every positive source also appears in another question):", {c: cov_count[c] for c in CLASSES})

    qB, qT = quotas(args.n_build, mix), quotas(args.n_test, mix)
    weakened = False
    for attempt in (1, 2):
        B, T, corpus, stats = select(args, Q, ents, q_img, q_txt, q_all, src_to_q, pos_of, ent_to_q, coverable, qB, qT, args.seed)
        cB, cT = Counter(Q[i]["cls"] for i in B), Counter(Q[i]["cls"] for i in T)
        short_B = {c: qB[c] - cB[c] for c in CLASSES if cB[c] < qB[c]}
        short_T = {c: qT[c] - cT[c] for c in CLASSES if cT[c] < qT[c]}
        if attempt == 1 and args.fill_shortfall == "image" and (short_B or short_T):
            for c, n in short_B.items():
                qB[c] -= n; qB["image"] += n
            for c, n in short_T.items():
                qT[c] -= n; qT["image"] += n
            weakened = True
            print("shortfall found; --fill-shortfall image: moving unfilled quota to the image class (WEAKENED)")
            continue
        break

    # ---- hard guarantees
    assert not (set(B) & set(T)), "build/test overlap"
    for i in T:
        assert set(Q[i]["pos"]) <= corpus, "test evidence missing from corpus"
    assert not ({Q[i]["qid"] for i in B} & {Q[i]["qid"] for i in T})
    assert not ({Q[i]["qnorm"] for i in B} & {Q[i]["qnorm"] for i in T}), "identical question text in both sets"

    # ---- outputs
    out = Path(args.out_dir)
    (out / "graph_corpus").mkdir(parents=True, exist_ok=True)
    img_srcs = sorted(s for s in corpus if s.startswith("img:"))
    txt_srcs = sorted(s for s in corpus if s.startswith("txt:"))
    with open(out / "graph_corpus" / "text_manifest.jsonl", "w", encoding="utf-8") as f:
        for s in img_srcs:
            x = sources[s]
            f.write(json.dumps({"doc_id": x["title"], "passage_id": f"cap_{x['image_id']}", "text": clean_caption(x["caption"] or x["title"]), "source_url": x["url"]}, ensure_ascii=False) + "\n")
        for s in txt_srcs:
            x = sources[s]
            f.write(json.dumps({"doc_id": x["title"], "passage_id": f"txt_{x['snippet_id']}", "text": x["text"], "source_url": x["url"]}, ensure_ascii=False) + "\n")
    with open(out / "graph_corpus" / "image_manifest.jsonl", "w", encoding="utf-8") as f:
        for s in img_srcs:
            x = sources[s]
            f.write(json.dumps({"image_id": x["image_id"], "doc_id": x["title"], "passage_ids": [f"cap_{x['image_id']}"],
                                "source_url": x["imgUrl"], "commons_url": x["url"]}, ensure_ascii=False) + "\n")
    with open(out / "graph_corpus" / "images_to_download.jsonl", "w", encoding="utf-8") as f:
        for s in img_srcs:
            x = sources[s]
            f.write(json.dumps({"image_id": x["image_id"], "imgUrl": x["imgUrl"], "commons_url": x["url"], "title": x["title"]}, ensure_ascii=False) + "\n")
    json.dump({"image_ids": [sources[s]["image_id"] for s in img_srcs], "text_snippet_ids": [sources[s]["snippet_id"] for s in txt_srcs]},
              open(out / "graph_corpus" / "source_ids.json", "w"), indent=0)

    def doc(s):
        x = sources[s]
        return ({"id": x["image_id"], "type": "image", "title": x["title"], "caption": x["caption"], "url": x["url"], "imgUrl": x["imgUrl"]}
                if x["type"] == "img" else {"id": x["snippet_id"], "type": "text", "title": x["title"], "text": x["text"], "url": x["url"]})

    def sid_out(s):
        return sources[s]["image_id"] if s.startswith("img:") else sources[s]["snippet_id"]

    build_gold = {s for i in B for s in Q[i]["pos"]}
    
    def entry(i, is_test=False):
        q = Q[i]

        e = {
            "qid": q["qid"],
            "question": q["question"],
            "answers": q["answers"],
            "Qcate": q["qcate"],
            "topic": q["topic"],
            "evidence_class": q["cls"],

            # Gold source IDs
            "gold_image_ids": [
                sid_out(s) for s in q["pos"] if s.startswith("img:")
            ],
            "gold_snippet_ids": [
                sid_out(s) for s in q["pos"] if s.startswith("txt:")
            ],

            # Complete positive documents
            "gold_documents": [
                doc(s) for s in q["pos"]
            ],

            # Distractor source IDs
            "distractor_image_ids": [
                sid_out(s) for s in q["neg"] if s.startswith("img:")
            ],
            "distractor_snippet_ids": [
                sid_out(s) for s in q["neg"] if s.startswith("txt:")
            ],

            # Complete negative documents, including image captions,
            # image URLs, text snippets, titles and source URLs
            "distractor_documents": [
                doc(s) for s in q["neg"]
            ]
        }

        if is_test:
            e["gold_presence_in_corpus"] = {
                sid_out(s): (
                    "build_gold" if s in build_gold
                    else "build_distractor_only"
                )
                for s in q["pos"]
            }

        return e

    
    with open(out / "build_qa.json", "w", encoding="utf-8") as f:
        json.dump(
            [entry(i) for i in B],
            f,
            ensure_ascii=False,
            indent=1
        )

    with open(out / "test_qa.json", "w", encoding="utf-8") as f:
        json.dump(
            [entry(i, is_test=True) for i in T],
            f,
            ensure_ascii=False,
            indent=1
        )


    # ---- report
    test_pos = {s for i in T for s in Q[i]["pos"]}
    gold_img = {s for i in B + T for s in Q[i]["pos"] if s.startswith("img:")}
    rng = random.Random(args.seed + 7)
    rb = []
    for c in CLASSES:
        rb += rng.sample(all_by_cls[c], min(qB[c], len(all_by_cls[c])))
    rand_corpus = {s for i in rb for s in Q[i]["S"]}
    report = {
        "weakened_by_fill_shortfall": weakened, "ner_mode": args.ner, "seed": args.seed,
        "evidence_classes_in_file": {c: cls_count[c] for c in CLASSES},
        "test_eligible_by_class": {c: cov_count[c] for c in CLASSES},
        "targets": {"build": qB, "test": qT},
        "selected": {"build": {c: cB[c] for c in CLASSES}, "test": {c: cT[c] for c in CLASSES}, "build_total": len(B), "test_total": len(T)},
        "shortfall": {"build": short_B, "test": short_T},
        "selection_stats": dict(stats),
        "corpus": {"images": len(img_srcs), "text_snippets": len(txt_srcs)},
        "connectivity_selected_corpus": connectivity(corpus, ents, args.max_df_corpus, gold_img),
        "connectivity_random_baseline_same_quotas": connectivity(rand_corpus, ents, args.max_df_corpus, gold_img),
        "source_overlap": {
            "build_test_question_id_overlap": 0, "build_test_question_text_overlap": 0,
            "test_positive_sources": len(test_pos),
            "test_positives_that_are_also_build_positives": len(test_pos & build_gold),
            "test_positives_present_only_as_build_distractors": len(test_pos - build_gold),
            "test_positives_missing_from_corpus": len(test_pos - corpus),
            "corpus_sources_that_are_test_positives": len(test_pos & corpus),
            "build_gold_sources": len(build_gold),
        },
    }
    image_connectivity = report["connectivity_selected_corpus"].get("images_in_giant_component")
    report["connectivity_target"] = {
        "target_images_in_giant_component": args.target_image_connectivity,
        "achieved_images_in_giant_component": image_connectivity,
        "target_met": image_connectivity is not None and image_connectivity >= args.target_image_connectivity,
        "note": "This is a selection heuristic; the target is measured after selection and is not guaranteed."
    }
    json.dump(report, open(out / "report.json", "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("targets", "selected", "shortfall", "corpus", "source_overlap")}, indent=2))
    print("connectivity, selected corpus:", json.dumps(report["connectivity_selected_corpus"]))
    print("connectivity target:", json.dumps(report["connectivity_target"]))
    print("connectivity, random baseline:", json.dumps(report["connectivity_random_baseline_same_quotas"]))
    if short_B or short_T:
        print("\nSHORTFALL: criteria were not weakened; see report.json -> shortfall")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
