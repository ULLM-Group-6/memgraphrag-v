# MemGraphRAG-V

Extending [MemGraphRAG](https://arxiv.org/abs/2606.00610) to natively index and retrieve **images**, reusing parts of [MG²-RAG](https://arxiv.org/abs/2604.04969).

2AMM20 Research Topics in Data Mining (TU/e) group project · 9-page ACM paper due **23 Oct 2026**.

**Status (3 Oct 2026):** research and planning only — no code yet. Both upstream codebases have been audited with stubbed models. A [methodology audit](docs/audit/methodology-audit-2026-10-03.md) (3 Oct, pending group review) proposes changes to data, retrieval and hypotheses. The outcome of the 2 Oct Snellius gate is not recorded here yet.

## The idea in one paragraph

Both base systems are forks of HippoRAG: they build a graph of entities and passages and rank passages with Personalized PageRank (PPR) from *seed* nodes matched to the question. We keep MemGraphRAG's text pipeline (schema/fact/passage memory with conflict resolution) unchanged, then attach image nodes and SAM3 object crops to its entity nodes. Our question is **which entities to ground**: entity *types* from MemGraphRAG's LLM-built memory (mapped to short noun phrases for SAM3) versus MG²-RAG's six named-entity labels — this controlled comparison is our main contribution. (MemGraphRAG's schema prompt is seeded with the same OntoNotes labels as spaCy, so we measure what types it really produces; see the audit.) At query time, CLIP matches of the query image become extra PPR seeds, combined with the text seeds after normalising each channel (`s = (1−λ)·ŝ_text + λ·ŝ_visual`). The main baselines are generic and entity-aware captioning of every image (query and knowledge base) indexed as text. Details: [docs/method.md](docs/method.md).

## Getting started

1. Read [docs/method.md](docs/method.md) (15 min), then the verdict and decisions of the [3 Oct methodology audit](docs/audit/methodology-audit-2026-10-03.md), then skim the summary of the original [audit report](docs/audit/audit-report.md) ([PDF](docs/audit/audit-report.pdf)).
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
│   ├── audit-report.md       full paper-vs-code audit of both repos (28 Sep; errata at top)
│   ├── audit-report.pdf      same, typeset (without errata)
│   ├── methodology-audit-2026-10-03.md   proposed changes + decisions for the group
│   ├── research-findings-2026-10-03.md   related work, datasets, models, prices (primary sources)
│   └── evidence/             stub-run summaries, HF model/dataset sizes, LLM cost estimate
└── tasks/                    one guide per workstream (see table above)
```

## Upstream

| | Paper | Code | Licence |
|---|---|---|---|
| MemGraphRAG | KDD 2026 · [arXiv 2606.00610](https://arxiv.org/abs/2606.00610) | [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) | MIT |
| MG²-RAG | ECCV 2026 · [arXiv 2604.04969](https://arxiv.org/abs/2604.04969) | [Daboolu/MG2-RAG](https://github.com/Daboolu/MG2-RAG) | MIT (its vendored SAM3 copy lacks SAM's licence file; the SAM License requires acknowledging SAM in publications) |

## Ground rules

- **No secrets in this repo** (API keys, Snellius reservation names, credentials). It is public.
- **Don't commit** papers' PDFs, model checkpoints, datasets or images — link to them.
- Tag claims like the audit does: **[V-run]** verified by running, **[V-read]** verified by reading code/paper, **[I]** inferred.
- Ask the group before large downloads (> a few GB) or paid API runs; the cost/size table is in [audit §3.3](docs/audit/audit-report.md#33-download-and-cost-plan-waiting-for-your-go-ahead).
