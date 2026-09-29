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

## Questions the plan doesn't answer yet

- Who owns what? (loaders + metrics, visual layer, baselines, writing)
- One chunk per document, or keep MemGraphRAG's 256-token windows? (affects image↔passage links and gold mapping)
- How is λ chosen without tuning on the test set? Need a small dev split.
- MLLM or text LLM for final answers? Same generator for every system?
- What do we do if SAM3 access isn't granted by ~6 Oct?

Write answers inline under each question (with your name and date) or open an issue per item.
