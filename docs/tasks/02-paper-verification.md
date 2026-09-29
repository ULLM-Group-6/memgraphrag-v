# Task 2: Verify and explore each paper on its own

**Goal:** independently check the audit's paper-vs-code findings, and understand each paper's method well enough to write our Background section. Split: one or two people per paper.

**Done when:** every row in the checklists below is marked ✅ confirmed, ❌ wrong (with correction), or ➕ extended, and each paper has a one-page summary in `docs/papers/<name>.md`.

> Don't take the audit at its word. Its claims are tagged [V-run] / [V-read] / [I]; the [I] ones are the most likely to be wrong. Line numbers refer to the audit-time commits — re-locate them in your clone.

## Sources

| Paper | arXiv | Code |
|---|---|---|
| MemGraphRAG | [2606.00610](https://arxiv.org/abs/2606.00610) | [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) |
| MG²-RAG | [2604.04969](https://arxiv.org/abs/2604.04969) | [Daboolu/MG2-RAG](https://github.com/Daboolu/MG2-RAG) |

Full claim tables: [audit §2.1 and §2.2](../audit/audit-report.md#2-paper-vs-code).

## MemGraphRAG checklist

| # | Claim to check | Where | Status |
|---|---|---|---|
| M1 | No agent framework: stages are sequential batch LLM passes | `src/MemGraphRAG.py` `index_with_memory` | |
| M2 | One ontology/schema LLM call **per fact** (cost driver) | `MemGraphRAG.py` ~369-425 | |
| M3 | Default `percentile` filter drops lowest 20 % of schemas regardless of frequency (paper: Freq ≥ τ) | ~452-485; `code/index.py:58-60` | |
| M4 | Conflict candidates are symbolic only; no embedding similarity (paper Eq. 5) | ~496-527 | |
| M5 | Modified triples are not re-normalised → duplicate entity nodes | ~756 | |
| M6 | Schema layer is never embedded or retrieved (paper Stage I) | `src/Memory.py:42` | |
| M7 | Type nodes get zero PPR seed mass (paper Eq. 7 missing) | ~1891-1892 | |
| M8 | No IDF term in passage seeds (paper Eq. 8 missing) | ~2160-2166 | |
| M9 | No Str-Acc / LLM-Acc; QA runner computes no metrics | `src/evaluation/`, `code/retrieval_dataset_test.py` | |
| M10 | G-Novel dataset not bundled; corpora re-chunked into 256-token windows (gold passage boundaries lost) | `dataset/`, `code/index.py` | |
| M11 | Which LLM, embedder and hyper-parameters produced the paper's main table? Are they reproducible from the repo? | paper §5 / appendix | |

## MG²-RAG checklist

| # | Claim to check | Where | Status |
|---|---|---|---|
| G1 | SAM3 prompts restricted to 6 spaCy labels (paper: all entities) | `MMGraphRAG.py:578-586` | |
| G2 | Showcase grounds "impala" only because spaCy labelled it PERSON | `examples/showcase/impala/text_structure.json` | |
| G3 | Showcase not produced from bundled input (mentions Kaokoland/Namibia) | same file vs `examples/data/impala_demo/input.json` | |
| G4 | EVA-CLIP text truncated to 77 tokens; entity embeddings computed but unused at retrieval | `eva_clip.py:98`; `MMGraphRAG.py:1341-1360` | |
| G5 | Seed aggregation deviates from Eq. 2/3 (doc-frequency division, sum not mean for objects) | ~885-1008 | |
| G6 | PPR is CuPy-only, no CPU path | ~1487-1622 | |
| G7 | No loaders for InfoSeek/ScienceQA/CrisisMMD; no metrics | `examples/`, `prompts/templates/` | |
| G8 | How big is each benchmark KB in the paper, and how were the 5k-document subsets built? (App. A.2) | paper | |
| G9 | Which numbers in the paper could we realistically reproduce on a MIG slice? | paper Table 1 / runtimes | |

## Explore beyond the checklist

- Write down each paper's **exact** equations for seeds and PPR in one notation. We need this for our Method section.
- For MemGraphRAG: find a question type where conflict resolution *should* matter. Does any bundled dataset contain real conflicts?
- For MG²-RAG: which of its ablations show the biggest gain from region-level (SAM3) evidence vs whole-image evidence? That tells us how much ontology-guided grounding can plausibly add.
- Note any claim in either paper without an ablation or error bars.

Output: `docs/papers/memgraphrag.md` and `docs/papers/mg2rag.md` (summary, equations, what's verified, open questions).
