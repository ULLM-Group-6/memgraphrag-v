# MemGraphRAG-V

Extending [MemGraphRAG](https://arxiv.org/abs/2606.00610) to natively index and retrieve **images**, reusing parts of [MG²-RAG](https://arxiv.org/abs/2604.04969).

2AMM20 Research Topics in Data Mining (TU/e) group project · 9-page ACM paper due **23 Oct 2026**.

**Status (29 Sep 2026):** research and planning only — no code yet. Both upstream codebases have been audited with stubbed models; nothing has been run on real models or Snellius yet.

## The idea in one paragraph

Both base systems are forks of HippoRAG: they build a graph of entities and passages and rank passages with Personalized PageRank (PPR) from *seed* nodes matched to the question. We keep MemGraphRAG's text pipeline (schema/fact/passage memory with conflict resolution) unchanged, then attach image nodes and SAM3 object crops to its entity nodes. Which entities get grounded in images is decided by MemGraphRAG's **schema types** (e.g. animals, buildings) rather than MG²-RAG's six named-entity labels — this is our main contribution. At query time, CLIP matches between the question and images become extra PPR seeds (`s = normalise(s_text + λ·s_visual)`). The main baseline is captioning every image and indexing captions as text. Details: [docs/method.md](docs/method.md).

## Getting started

1. Read [docs/method.md](docs/method.md) (15 min), then skim the summary of the [audit report](docs/audit/audit-report.md) ([PDF](docs/audit/audit-report.pdf)).
2. Pick a workstream:

| # | Workstream | Guide | Needs |
|---|---|---|---|
| 1 | Snellius smoke test — a real MemGraphRAG run (gate: **2 Oct**) | [tasks/01-snellius-smoke-test.md](docs/tasks/01-snellius-smoke-test.md) | Snellius account, LLM API key or vLLM |
| 2 | Verify and explore each paper's methodology on its own | [tasks/02-paper-verification.md](docs/tasks/02-paper-verification.md) | The papers + a clone of each repo |
| 3 | Probe the project plan and its assumptions | [tasks/03-plan-probing.md](docs/tasks/03-plan-probing.md) | method.md, audit report |
| 4 | Find missed information and published papers | [tasks/04-literature-search.md](docs/tasks/04-literature-search.md) | Scholar / arXiv |

3. Record findings in the task file or a new doc under `docs/`, and open an issue for anything that contradicts the audit or the plan.

## Repo map

```
docs/
├── method.md                 proposed method, evaluation, timeline, caveats
├── audit/
│   ├── audit-report.md       full paper-vs-code audit of both repos (with evidence tags)
│   ├── audit-report.pdf      same, typeset
│   └── evidence/             stub-run summaries, HF model/dataset sizes, LLM cost estimate
└── tasks/                    one guide per workstream (see table above)
```

## Upstream

| | Paper | Code | Licence |
|---|---|---|---|
| MemGraphRAG | [arXiv 2606.00610](https://arxiv.org/abs/2606.00610) | [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) | MIT |
| MG²-RAG | [arXiv 2604.04969](https://arxiv.org/abs/2604.04969) | [Daboolu/MG2-RAG](https://github.com/Daboolu/MG2-RAG) | MIT (vendored SAM3 has its own licence) |

## Ground rules

- **No secrets in this repo** (API keys, Snellius reservation names, credentials). It is public.
- **Don't commit** papers' PDFs, model checkpoints, datasets or images — link to them.
- Tag claims like the audit does: **[V-run]** verified by running, **[V-read]** verified by reading code/paper, **[I]** inferred.
- Ask the group before large downloads (> a few GB) or paid API runs; the cost/size table is in [audit §3.3](docs/audit/audit-report.md#33-download-and-cost-plan-waiting-for-your-go-ahead).
