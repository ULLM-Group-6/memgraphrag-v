# Proposed method: MemGraphRAG-V

**Status:** proposal as of 29 Sep 2026, not yet validated on real models. Challenge it: see [tasks/03-plan-probing.md](tasks/03-plan-probing.md).

We extend MemGraphRAG with images by adding a visual layer *after* text indexing (late fusion). SAM3 object crops are chosen using MemGraphRAG's entity **types**, and image matches enter retrieval as extra Personalized PageRank (PPR) seeds. The main baseline is "caption every image and index the captions as text". The paper is due **23 Oct 2026**.

Evidence behind every claim here is in the [audit report](audit/audit-report.md). Tags there: **[V-run]** verified by running, **[V-read]** verified by reading code/paper, **[I]** inferred.

## Background in brief

- **Embeddings** turn text or images into vectors; similar meaning gives nearby vectors.
- **RAG** fetches relevant passages and puts them in the LLM prompt, so the LLM answers from our documents.
- **Graph RAG (the HippoRAG family, which both repos come from).** At indexing time an LLM extracts facts as triples, e.g. `(Film X, directed by, Person Y)`. Entities and passages become graph nodes.
- **Personalized PageRank** ranks nodes by where a random walker ends up. It starts at *seed* nodes matched to the question; at each step it either follows an edge or jumps back to a seed. Passages connected to several seeds score high, which is what solves multi-hop questions.

```
p = α·s + (1 − α)·Pᵀ·p
```

`s` is the seed vector (where the walker restarts), `P` is how it moves along edges, `p` is every node's final score. **Key idea: anything that can become a seed can influence the ranking.** So we add images as new seeds instead of writing a new retrieval algorithm.

## What each system contributes

| System | Paper | Code | What it adds | What we take |
|---|---|---|---|---|
| MemGraphRAG | arXiv 2606.00610 | [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) | Three-layer memory (schema types → facts → passages); detects and resolves conflicting facts during indexing | The whole text pipeline, unchanged |
| MG²-RAG | arXiv 2604.04969 | [Daboolu/MG2-RAG](https://github.com/Daboolu/MG2-RAG) | Image nodes; SAM3 object crops linked to entities; EVA-CLIP embeddings putting text and images in one space; GPU PPR | Image nodes, crop nodes, CLIP store, seed-fusion logic, SAM3 wrapper |

Both are HippoRAG forks. They share MD5 node IDs (`chunk-`/`entity-` prefixes via `compute_mdhash_id`), an identical `text_processing` normaliser, parquet `EmbeddingStore`s, igraph, and PPR over passage nodes [V-read]. The same entity therefore gets the same node ID in both graphs, so graphs can be merged by ID.

## Pipeline

**Indexing (once per corpus)**

1. Run MemGraphRAG as released: OpenIE → per-fact schema extraction → ontology filter → conflict detection → conflict resolution → memory graph (type, entity, passage nodes).
2. Add one **image node** per image, linked to its passage/document.
3. **Ontology-guided grounding (our contribution).** Prompt SAM3 with surviving entities whose *schema type* describes something visible (animal, building, artwork, object…). Link each crop to its entity node.
4. Embed images and crops with a CLIP-style encoder (SigLIP2 for development, EVA-CLIP-8B optionally for a final run), stored separately from the bge text vectors.

**Retrieval (per question)**

1. Text seeds exactly as MemGraphRAG does now (facts and passages via bge).
2. **Visual seeds:** embed the question (text, plus image if any) with CLIP; nearest image and crop nodes become seeds.
3. Combine and run PPR on the merged graph:

```
s = normalise(s_text + λ · s_visual)
```

4. Top passages (and their images) go into the LLM/MLLM prompt.

## Design choices and why

1. **Late fusion.** Text and images meet only in the seed vector. We never compare a bge text vector with a CLIP image vector (different spaces). CLIP text encoders also truncate at 77 tokens (EVA-CLIP, `eva_clip.py:98` [V-read]), making them poor passage encoders. The text pipeline stays intact and regression-testable.
2. **Ontology-guided grounding.** MG²-RAG only prompts SAM3 with entities whose spaCy label is FAC, LOC, ORG, PERSON, PRODUCT or WORK_OF_ART (`MMGraphRAG.py:578-586` [V-read]). Its own impala showcase works only because spaCy mislabelled "Impala" as PERSON; with correct labels the same demo grounds **zero** crops [V-run, stubbed]. MemGraphRAG's schema layer says what *kind* of thing an entity is — the right signal for "can this appear in a picture?".
3. **Build on MemGraphRAG, not MG²-RAG.** The brief is to extend MemGraphRAG. MG²-RAG's text side is spaCy + dependency rules (weaker than LLM extraction), needs 2 GPUs + CUDA + CuPy by default, and ships no loaders or metrics.
4. **Captioning is the baseline, not the method.** Captioning images into text (audit Option D) is what both papers argue against, and what a reviewer will ask about first.

Alternatives considered (full table: audit §5): B multimodal memory with MLLM conflict verification (3–4 person-weeks, high risk); C MG²-RAG as base (story becomes "improving MG²-RAG"); E single CLIP encoder for everything (useful as an ablation, likely text regression).

## Evaluation

| ID | Hypothesis | Compared against |
|---|---|---|
| H1 | Visual seeds improve retrieval on image-dependent questions | Text-only MemGraphRAG |
| H2 | The visual layer beats captioning images into text | Caption MemGraphRAG |
| H3 | Ontology-guided prompts ground more entities than NER-label prompts, and that improves retrieval | MG²-RAG-style prompting |
| H4 | Text-only performance does not drop | Text-only MemGraphRAG |

| Run | Role |
|---|---|
| MemGraphRAG-V (full) | Our method |
| Caption MemGraphRAG (Qwen2.5-VL captions) | Main baseline |
| Text-only MemGraphRAG (as released) | Baseline |
| Dense CLIP retrieval | Baseline |
| MG²-RAG | Baseline, if it runs on Snellius |
| No ontology guidance (NER-label prompts) | Ablation |
| No visual seeds (λ = 0) | Ablation |
| No memory (legacy `index()` path) | Ablation |
| λ sweep; single-encoder swap (Option E) | Ablations if time allows |

**Metrics** (all must be written by us — MemGraphRAG's QA runner computes none, MG²-RAG ships none): doc-ID Recall@{1,5,10} and MRR; containment accuracy and LLM-judge accuracy (the MemGraphRAG paper's metrics); grounded-entity counts for H3.

**Data:** ~500 questions sampled from M2KR InfoSeek/E-VQA test splits (≈0.1 GB text, MIT), fetching only the images those questions need. The full image sources (OVEN 294 GB, AToMiC 174 GB) are gated and out of scope. Text regression on a subsample of MemGraphRAG's bundled HotpotQA/2Wiki/MuSiQue (a full corpus costs an estimated $13–37 with gpt-4o-mini; see [evidence/indexing_cost_estimate.json](audit/evidence/indexing_cost_estimate.json)).

## Timeline (proposed)

| Dates | Milestone |
|---|---|
| 29 Sep – 2 Oct | Setup and Snellius smoke test. **Gate (2 Oct): a real MemGraphRAG run works.** |
| 5 – 9 Oct | Loaders, metrics, caption baseline |
| 5 – 13 Oct | Visual layer: image nodes, ontology-guided SAM3, CLIP store, visual seeds |
| 12 – 16 Oct | Main runs and ablations. **Results freeze 16 Oct.** |
| 5 – 22 Oct | Paper writing (background and method start early) |
| 23 Oct | Submission (9-page ACM) |

If the week-1 gate slips, drop MG²-RAG as a baseline before dropping any ablation. (The audit's §8 has an earlier, slightly different plan; this table supersedes it.)

## Caveats

| Caveat | Status | Fallback |
|---|---|---|
| Integration only shown with stubbed LLMs/embedders | [V-run] mocked end-to-end | Real smoke test in week 1 |
| MemGraphRAG code lacks schema retrieval, type-node seeding (Eq. 7), IDF term (Eq. 8); type nodes get zero seed mass | [V-read, V-run] | Call our baseline "MemGraphRAG as released"; implementing Eq. 7–8 is ~1 day if wanted |
| MemGraphRAG launch scripts crash; README paths wrong; `gritlm` is a hard import | [V-run] | Our own launch commands (see smoke-test task) |
| Default ontology filter dropped 2 correct facts; modified triples create duplicate entity nodes | [V-run, stub data] | Report as limitation; try `absolute` filter mode |
| SAM3 needs manual Hugging Face approval | Pending | Request now; fallback OWLv2 / GroundingDINO or image-level only |
| EVA-CLIP-8B is 33 GB download, ~16 GB VRAM fp16 | [V-run sizes, I VRAM] | SigLIP2-so400m (4.5 GB) for development |
| InfoSeek/E-VQA are mostly single-hop; memory may add little | [I] | Prefer E-VQA two-hop items; report R@k and QA |
| Windows igraph cannot read GraphML | [V-run] | Run everything on Snellius (Linux) |

Every expected gain above is a hypothesis; nothing has been measured on real models yet.
