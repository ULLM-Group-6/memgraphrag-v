"""Show a retrieval run as a page of thumbnails, to eyeball or demo it.

For every question: the question, its gold document, and the retrieved
documents in rank order with their image, score and first passage. Prints
one line per question and writes a self-contained runs/<id>/report.html
(thumbnails are embedded, so the file can be copied anywhere and opened).
    python scripts/show_retrieval.py --run-id dummy-direct
"""

import argparse
import base64
import html
import io
from pathlib import Path

from PIL import Image

from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.io import read_jsonl
from memgraphrag_v.paths import Artifacts
from memgraphrag_v.schemas import ImageRecord, PassageRecord, QuestionRecord, RetrievalResult

THUMB = 160
SNIPPET = 160

STYLE = """
:root { --bg: #fafaf9; --card: #fff; --text: #1c1917; --muted: #78716c; --line: #e7e5e4;
        --hit: #15803d; --miss: #b91c1c; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1c1917; --card: #292524; --text: #f5f5f4; --muted: #a8a29e; --line: #44403c;
          --hit: #4ade80; --miss: #f87171; } }
body { background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, sans-serif;
       margin: 0 auto; padding: 16px; max-width: 1000px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.summary { color: var(--muted); margin-bottom: 16px; }
.q { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px;
     margin-bottom: 12px; }
.q h2 { font-size: 15px; margin: 0 0 2px; }
.meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
.hit { color: var(--hit); font-weight: 600; } .miss { color: var(--miss); font-weight: 600; }
.docs { display: flex; gap: 8px; overflow-x: auto; }
.doc { flex: 0 0 168px; border: 2px solid transparent; border-radius: 6px; padding: 4px; }
.doc.gold { border-color: var(--hit); }
.doc img { width: 160px; height: 160px; object-fit: contain; background: #fff; border-radius: 4px; }
.doc b { display: block; } .doc small { color: var(--muted); display: block; }
"""


def thumbnail(path: Path) -> str:
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return ""
    image.thumbnail((THUMB, THUMB))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def gold_rank(result: RetrievalResult, gold: set[str]) -> int | None:
    return next((rank for rank, e in enumerate(result.evidence, 1) if e.doc_id in gold), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    artifacts = Artifacts.from_env(ExperimentConfig.load(args.config).artifacts_root)
    run_dir = artifacts.run_dir(args.run_id)
    results = read_jsonl(run_dir / "retrieval.jsonl", RetrievalResult)
    questions = {q.qid: q for q in read_jsonl(artifacts.questions, QuestionRecord)}
    images = {i.image_id: i for i in read_jsonl(artifacts.image_manifest, ImageRecord)}
    passages = {p.passage_id: p.text for p in read_jsonl(artifacts.text_manifest, PassageRecord)}

    ranks = [gold_rank(r, set(questions[r.question_id].gold_doc_ids)) for r in results]
    at_1, in_top = sum(r == 1 for r in ranks), sum(r is not None for r in ranks)
    summary = f"{len(results)} questions, gold document first for {at_1}, retrieved at all for {in_top}"

    cards = []
    for result, rank in zip(results, ranks):
        q = questions[result.question_id]
        found = "not retrieved" if rank is None else f"gold at rank {rank}"
        print(f"{'ok  ' if rank == 1 else 'MISS'} {q.question}  ->  "
              + ", ".join(f"{e.doc_id} ({e.score:.3f})" for e in result.evidence))
        docs = []
        for e in result.evidence:
            src = thumbnail(artifacts.image_path(images[e.image_id])) if e.image_id else ""
            text = passages.get(e.passage_ids[0], "") if e.passage_ids else ""
            text = text[:SNIPPET] + ("…" if len(text) > SNIPPET else "")
            docs.append(
                f'<div class="doc{" gold" if e.doc_id in q.gold_doc_ids else ""}">'
                + (f'<img src="{src}" alt="">' if src else "")
                + f"<b>{html.escape(e.doc_id)}</b><small>score {e.score:.3f}</small>"
                + f"<small>{html.escape(text)}</small></div>"
            )
        cards.append(
            f'<div class="q"><h2>{html.escape(q.question)}</h2>'
            f'<div class="meta">{html.escape(q.qid)} · {q.split} · gold: {html.escape(", ".join(q.gold_doc_ids))} · '
            f'<span class="{"hit" if rank == 1 else "miss"}">{found}</span></div>'
            f'<div class="docs">{"".join(docs)}</div></div>'
        )

    out = run_dir / "report.html"
    out.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Retrieval {html.escape(args.run_id)}</title><style>{STYLE}</style></head><body>"
        f"<h1>Run {html.escape(args.run_id)}</h1><div class=\"summary\">{summary}</div>"
        f'{"".join(cards)}</body></html>',
        encoding="utf-8",
    )
    print(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
