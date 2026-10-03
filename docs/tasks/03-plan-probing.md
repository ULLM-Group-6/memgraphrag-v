# Task 3: Probe the project plan

**Goal:** try to break the [proposed method](../method.md) before we spend compute on it. Every assumption below was made by one person in one audit; treat it as a claim to test, not a decision.

**Done when:** each question has a written answer (or "open, blocks X") and any change to the plan is reflected in `docs/method.md` via a PR.

## Assumptions to challenge

| # | Assumption | Why it matters | How to test it |
|---|---|---|---|
| P1 | Entity IDs from MemGraphRAG and MG²-RAG coincide, so graphs merge by ID | Whole integration rests on it | Hash the same entity string with both `compute_mdhash_id` + `text_processing`; check MemGraphRAG's resolved triples go through the same normaliser |
| P2 | Seed-level late fusion (`s_text + λ s_visual`) is enough; no shared embedding space needed | Core design choice | Sketch a question where the answer requires comparing a text vector with an image vector. Does PPR still get there? |
| P3 | MemGraphRAG schema types reliably signal "visually groundable" | Our main novelty | Inspect `initial_memory_with_schema.json` from the smoke test: what types come out? Are they fine-grained enough (e.g. `<ANIMAL>` vs `<ENTITY>`)? |
| P4 | Image-only questions reach the graph | MemGraphRAG falls back to dense retrieval when no fact passes the threshold (`MemGraphRAG.py:1100-1105`) | Decide how the dispatch changes when a question has an image |
| P5 | InfoSeek / E-VQA have enough multi-hop structure for memory to matter | If not, H1–H3 may show nothing | Sample 50 questions; label single- vs multi-hop; look at E-VQA two-hop subset |
| P6 | ~500 questions + only their images is feasible to build in a week | Data is the biggest schedule risk | Check M2KR fields: are image URLs/IDs present? Are the Wikipedia images still downloadable? |
| P7 | Captioning is a strong enough baseline to make a win meaningful | Reviewers will ask | Pick the captioner and prompt now; consider entity-aware captions as a stronger variant |
| P8 | SigLIP2 is an acceptable stand-in for EVA-CLIP-8B | Compute budget | Does MG²-RAG report encoder ablations? Is a different encoder a confound vs the MG²-RAG baseline? |
| P9 | Implementing the paper's missing Eq. 7/8 is not needed | Fairness of "MemGraphRAG" baseline | Decide how we name and describe the baseline in the paper |
| P10 | Budget: < US$50 of LLM calls and mostly free MIG hours | Feasibility | Recompute with smoke-test token counts (Task 1) × planned corpus sizes |
| P11 | The 2 Oct gate and 16 Oct freeze are realistic | Deadline 23 Oct | Owners estimate their own pieces; compare with the timeline |

## Answers from the 3 Oct methodology audit

Claude (agent), 3 Oct 2026, for group review. Details and sources are in the [methodology audit](../audit/methodology-audit-2026-10-03.md) (A#, C#, D#) and the [research findings](../audit/research-findings-2026-10-03.md). Challenge these too.

| # | Answer | Status |
|---|---|---|
| P1 | The normaliser and hash functions are identical, but Option A never imports MG²-RAG's graph, so cross-repo ID equality isn't load-bearing (C10). The real risk is *inside* MemGraphRAG: query-time lookup uses `.lower()`, not `text_processing`, and modified triples aren't re-normalised. | Less critical than stated |
| P2 | Seed fusion is workable only if each channel is normalised first (A4) and visual seeds bypass the fact gate (A3). | Plan changed |
| P3 | At risk. The schema prompt suggests the 18 OntoNotes labels, the same set as spaCy (A5). Measure with the type histogram in task 01. | **Open, blocks H3** |
| P4 | Not as specified. Deictic questions make the dense fallback the common case (A3). The dispatch must change. | **Blocking**; plan changed |
| P5 | No. M2KR's E-VQA split has no two-hop items, and KB-VQA is recognition-dominated (A2, A11). | Decision D1 |
| P6 | Query images: yes, ungated in `M2KR_Images`. KB images: **not in M2KR**; use MuKA URLs or the official E-VQA KB (A1, A19). | Plan changed |
| P7 | Only if it captions query *and* KB images and has an entity-aware variant (A8). | Plan changed |
| P8 | MG²-RAG has no encoder ablation; a different encoder is a confound (A12). | Decision D3 |
| P9 | Keep "MemGraphRAG (released code)", without Eq. 7–8 (D5). | Recommendation unchanged |
| P10 | gpt-4o-mini is still $0.15 / $0.60 per 1M tokens. With one index per corpus variant the total is ≈ $40–140, above $50, unless corpora are subsampled, batched or self-hosted (A14). | Re-check with smoke-test counts |
| P11 | The 2 Oct gate outcome isn't recorded. The proposed timeline adds data work (MuKA images, MMQA) on 4–8 Oct. | Owners to confirm |

Open questions below, partly answered:
- **Chunking:** one document (or section) per chunk (A13).
- **λ:** tune on a dev split.
- **Generator:** the same MLLM for every system (D4).
- **SAM3 by 6 Oct:** fall back to Grounding DINO / OWLv2 (D8).

## Questions the plan doesn't answer yet

- Who owns what? (loaders + metrics, visual layer, baselines, writing)
- One chunk per document, or keep MemGraphRAG's 256-token windows? (affects image↔passage links and gold mapping)
- How is λ chosen without tuning on the test set? Need a small dev split.
- MLLM or text LLM for final answers? Same generator for every system?
- What do we do if SAM3 access isn't granted by ~6 Oct?

Write answers inline under each question (with your name and date) or open an issue per item.
