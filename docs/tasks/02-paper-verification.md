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
| M8 | No IDF term in passage seeds (paper Eq. 8 missing) | ~2160-2166 | ➕ an upstream issue (#6, 28 Jul 2026, unanswered) asks the same question; still re-check in code |
| M9 | No Str-Acc / LLM-Acc; QA runner computes no metrics | `src/evaluation/`, `code/retrieval_dataset_test.py` | |
| M10 | G-Novel dataset not bundled; corpora re-chunked into 256-token windows (gold passage boundaries lost) | `dataset/`, `code/index.py` | ❌ partly: per-passage `dataset/*/*_corpus.json` with titles exist, so boundaries are lost only via `index.py` (3 Oct audit C4). G-Novel absence confirmed by upstream issue #7 |
| M11 | Which LLM, embedder and hyper-parameters produced the paper's main table? Are they reproducible from the repo? | paper §5 / appendix | ➕ NV-Embed-v2, top-k 5, gpt-4o-mini (index, answer, judge), temperature 0, 1,000 validation questions per dataset ([findings §3.4](../audit/research-findings-2026-10-03.md#34-what-mg-rag-and-memgraphrag-evaluated-on-q8)); repo defaults differ (bge, top-k 10) |

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
| G8 | How big is each benchmark KB in the paper, and how were the 5k-document subsets built? (App. A.2) | paper | ➕ 100k random documents from the ≈2M-page E-VQA KB; 5k subsets "ensuring inclusion of the necessary evidence"; Table 2 vs Table 6 use different KB sizes ([findings §3.4](../audit/research-findings-2026-10-03.md#34-what-mg-rag-and-memgraphrag-evaluated-on-q8)) |
| G9 | Which numbers in the paper could we realistically reproduce on a MIG slice? | paper Table 1 / runtimes | |

## Explore beyond the checklist

- Write down each paper's **exact** equations for seeds and PPR in one notation. We need this for our Method section.
- For MemGraphRAG: find a question type where conflict resolution *should* matter. Does any bundled dataset contain real conflicts?
- For MG²-RAG: which of its ablations show the biggest gain from region-level (SAM3) evidence vs whole-image evidence? That tells us how much ontology-guided grounding can plausibly add. *Partial answer (3 Oct):* there is no encoder ablation; "w/o MNF" (no text-entity ↔ region alignment) drops E-VQA (5k) R@1 57.8 → 43.8 and BEM 60.24 → 55.38 (MG² Table 6). See [findings §4.3](../audit/research-findings-2026-10-03.md#43-textimage-matching-and-encoders-q11).
- Note any claim in either paper without an ablation or error bars.

Output: `docs/papers/memgraphrag.md` and `docs/papers/mg2rag.md` (summary, equations, what's verified, open questions).
