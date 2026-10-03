# Proposed method: MemGraphRAG-V

**Status:** proposal, revised 3 Oct 2026 after the [methodology audit](audit/methodology-audit-2026-10-03.md) (pending group review). Not yet validated on real models. Challenge it: see [tasks/03-plan-probing.md](tasks/03-plan-probing.md).

We extend MemGraphRAG with images by adding a visual layer *after* text indexing (late fusion). The question we study is **which entities to ground**. SAM3 object crops are chosen using entity *types* from MemGraphRAG's LLM-built memory, rather than MG²-RAG's named-entity labels. We then ask whether native visual evidence, entering retrieval as extra Personalized PageRank (PPR) seeds, beats captioning images into text. The paper is due **23 Oct 2026**.

Evidence behind every claim here is in the [audit report](audit/audit-report.md) (28 Sep) and the [3 Oct research findings](audit/research-findings-2026-10-03.md). Tags: **[V-run]** verified by running, **[V-read]** verified by reading code/paper, **[I]** inferred.

## Background in brief

- **Embeddings** turn text or images into vectors; similar meaning gives nearby vectors.
- **RAG** fetches relevant passages and puts them in the LLM prompt, so the LLM answers from our documents.
- **Graph RAG (the HippoRAG family, which both repos come from).** At indexing time an LLM extracts facts as triples, e.g. `(Film X, directed by, Person Y)`. Entities and passages become graph nodes.
- **Personalized PageRank** ranks nodes by where a random walker ends up. It starts at *seed* nodes matched to the question; at each step it either follows an edge or jumps back to a seed. Passages connected to several seeds score high, which is what solves multi-hop questions.

```
p = α·s + (1 − α)·Pᵀ·p
```

`s` is the seed vector (where the walker restarts), `P` is how it moves along edges, `p` is every node's final score. **Key idea: anything that can become a seed can influence the ranking.** So we add images as new seeds instead of writing a new retrieval algorithm. (MG²-RAG already fuses visual and text seeds before PPR; the seed mechanism is not our contribution.)

## What each system contributes

| System | Paper | Code | What it adds | What we take |
|---|---|---|---|---|
| MemGraphRAG | KDD 2026, arXiv 2606.00610 | [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) | Three-layer memory (schema types → facts → passages); detects and resolves conflicting facts during indexing. Its released retrieval is HippoRAG 2's seed rule on this graph. | The whole text pipeline, unchanged |
| MG²-RAG | ECCV 2026, arXiv 2604.04969 | [Daboolu/MG2-RAG](https://github.com/Daboolu/MG2-RAG) | Image nodes; SAM3 object crops linked to entities; EVA-CLIP embeddings putting text and images in one space; visual PPR seeds; GPU PPR | Image nodes, crop→entity seed logic, CLIP store, SAM3 usage |

Both are HippoRAG forks. They share MD5 node IDs (`chunk-`/`entity-` prefixes via `compute_mdhash_id`), an identical `text_processing` normaliser, parquet `EmbeddingStore`s, igraph, and PPR over passage nodes [V-read]. We do not import MG²-RAG's graph; grounding edges are computed against MemGraphRAG's own node IDs, so cross-repo ID equality is a convenience, not a requirement.

## Pipeline

**Indexing (once per corpus)**

1. Load **one document (or Wikipedia section) per chunk** and call `index_with_memory` directly. `code/index.py` slides 256-token windows across document boundaries, which loses the document↔image links and gold IDs.
2. Run MemGraphRAG as released: OpenIE → per-fact schema extraction → ontology filter → conflict detection → conflict resolution → memory graph (type, entity, passage nodes).
3. Add one **image node** per image, linked to its document's passage(s). Documents may have zero or several images.
4. **Type-guided grounding (our contribution).**
   - Decide once per distinct type whether it is *visual* (one LLM yes/no call per type, or a reviewed list; publish the list).
   - Map each visual type to a generic noun phrase ("animal", "plant", "building", "painting"). SAM3 is built for short noun phrases, not entity names.
   - Prompt SAM3 on each image with the noun phrases of the visual-typed entities in that image's passage, and link each crop to the matching entity.
   - Take candidates from the schema-annotated memory *before* the ontology filter, which otherwise deletes facts and their entities.
5. Embed images and crops with a CLIP-style encoder (SigLIP2 for development), stored separately from the bge text vectors. Crops are *not* graph nodes: a crop match seeds its linked entity, as in MG²-RAG.

**Retrieval (per question)**

1. Text seeds exactly as MemGraphRAG does now (facts and passages via bge).
2. **Visual seeds:** embed the **query image** (and, secondarily, the question text) with CLIP. Nearest image nodes seed themselves; nearest crops seed their linked entities.
3. Combine the two channels after normalising each to unit mass. MemGraphRAG's text seeds are unnormalised and their passage mass grows with corpus size, so a raw sum would make λ mean something different for every query.

```
s = (1 − λ) · s_text / ‖s_text‖₁  +  λ · s_visual / ‖s_visual‖₁
```

4. **Dispatch change:** compute visual seeds *before* MemGraphRAG's fact gate. If no text fact passes the threshold (the common case for "what is this plant…?" questions), run PPR on visual + passage seeds instead of falling back to dense passage retrieval (`MemGraphRAG.py:1100-1105`).
5. Top passages (and their images) go into the MLLM prompt.

## Design choices and why

1. **Late fusion.** Text and images meet only in the seed vector. We never compare a bge text vector with a CLIP image vector (different spaces). CLIP-style text encoders are also short-context: **64 tokens for SigLIP2**, 77 for EVA-CLIP-8B. That makes them poor passage encoders. The text pipeline stays intact and regression-testable.
2. **Type-guided grounding.** MG²-RAG prompts SAM3 only with spaCy entities labelled FAC, LOC, ORG, PERSON, PRODUCT or WORK_OF_ART (`MMGraphRAG.py:578-586` [V-read]). Its impala showcase works only because spaCy mislabelled "Impala" as PERSON; with correct labels it grounds **zero** crops [V-run, stubbed]. MemGraphRAG differs in two ways:
   - its entities come from LLM triples, so common nouns ("impala") become nodes;
   - each entity gets a type from context.

   **Caveat:** MemGraphRAG's schema prompt suggests the same 18 OntoNotes labels as spaCy, with no "animal" (`prompt.py:298-326` [V-read]). Whether the LLM produces finer types is measured in week 1 (task 01). If it doesn't, the type-guided arm uses the repo's unused fine-grained `entity_type_extraction` prompt.
3. **Build on MemGraphRAG, not MG²-RAG.** The brief is to extend MemGraphRAG. MG²-RAG's text side is spaCy + dependency rules (weaker than LLM extraction), needs 2 GPUs + CUDA + CuPy by default, and ships no loaders or metrics.
4. **Captioning is the baseline, not the method.** Captioning images into text (audit Option D) is what both papers argue against, and what a reviewer will ask about first. It must be a *strong* baseline (below), or a win means little.

Alternatives considered (full table: audit §5): B multimodal memory with MLLM conflict verification (3–4 person-weeks, high risk); C MG²-RAG as base (story becomes "improving MG²-RAG"); E single CLIP encoder for everything (useful as an ablation, likely text regression).

**Related work to position against** (details: [research findings §2](audit/research-findings-2026-10-03.md#2-novelty-assessment-and-related-work)):
- MG²-RAG: image nodes, crops linked to entities, visual PPR seeds.
- HVM-GraphRAG (2607.24861): conflict-resolving graph construction with images, via captions.
- RAG-Anything (2510.12323): caption-centric multimodal graph RAG.
- mKG-RAG (2508.05318): MLLM-built multimodal KG for KB-VQA.
- EviProp (2606.08979): visual PPR priors.

No paper found chooses grounding targets from LLM-derived entity types.

## Evaluation

| ID | Hypothesis | Compared against |
|---|---|---|
| H1 | Visual seeds improve retrieval over the best non-graph visual retrieval | Dense CLIP image→image/text retrieval; MG²-RAG. (vs text-only MemGraphRAG only as a sanity check: on KB-VQA it cannot know which entity "this plant" is) |
| H2 | Native visual evidence beats captioning images into text | Generic and entity-aware caption baselines, with and without query-image captions |
| H3 | Type-guided grounding selects better entities than NER-label filtering, and that improves retrieval | Same LLM entities under: no grounding / all entities / NER-label filter (MG²-style) / type-guided (+ fine-grained types if needed) |
| H4 | Text-only performance does not drop | Unmodified MemGraphRAG graph on text-only questions |

| Run | Role |
|---|---|
| MemGraphRAG-V (full) | Our method |
| Caption MemGraphRAG, generic captions of query + KB images | Main baseline |
| Caption MemGraphRAG, entity-aware captions of query + KB images | Main baseline (strong) |
| Text-only MemGraphRAG ("released code") | Baseline / H4 |
| Dense CLIP retrieval (image→image, image→text) | Baseline |
| Zero-shot MLLM (no retrieval) | Baseline |
| MG²-RAG, same encoder as ours | Baseline, if it runs on Snellius |
| Grounding arms for H3 (none / all entities / NER-label / type-guided) | Ablation |
| Visual graph, no visual seeds (λ = 0) | Ablation (not equal to text-only: image nodes change the walk) |
| No memory: legacy `index()` path (≈ HippoRAG 2 pipeline) | Ablation |
| λ sweep; single-encoder swap (Option E) | Ablations if time allows |

All systems use the same MLLM for captions and answers. Captions are indexed as separate passages linked to their document. Tune λ on a small dev split, never on test.

**Metrics** (all written by us: MemGraphRAG's QA runner computes none, MG²-RAG ships none):
- doc-ID Recall@{1,5,10} and MRR;
- containment accuracy and LLM-judge accuracy (the MemGraphRAG paper's metrics), plus BEM for E-VQA if TensorFlow installs on Snellius;
- grounding coverage and precision for H3 (entities grounded; share of crops that a manual check of a sample confirms);
- bootstrap confidence intervals on every comparison.

Never put our subset numbers in the same table as either paper's published numbers; every baseline is re-run on our data.

**Data** (proposal; final choice is decision D1 in the audit):

- **KB-VQA: M2KR E-VQA test.** Sample 300–500 questions. The KB is the gold pages plus distractor pages from M2KR's 19,267-page E-VQA test corpus (text, MIT). M2KR passages are **text-only**, so:
  - **KB images** come from MuKA's released Wikimedia URL lists (one image per entity, covering every M2KR E-VQA test passage).
  - **Query images** come from the ungated `BByrneLab/M2KR_Images` (iNat zip, readable per file, plus the 2.8 GB GLDv2 tar).

  M2KR's E-VQA split has **no two-hop questions** (1,000 templated + 2,750 automatic), so this set is single-hop and recognition-dominated. Two-hop items exist only in the official E-VQA `test.csv` with the official KB. InfoSeek is optional: M2KR's "test" is InfoSeek *validation*, and it adds a 9 GB image tar.
- **Images on the corpus side, text questions (proposed): MMQA dev.** 940 of 2,441 dev questions need images, 569 of them compositional. It has 57k Wikipedia entity images (2.36 GB) and gold doc IDs. This is where grounding and multi-hop can matter. Its licence is not stated: check before use.
- **Text regression** on a subsample of MemGraphRAG's bundled HotpotQA/2Wiki/MuSiQue, indexed per passage from `dataset/*/*_corpus.json` so gold titles map to doc IDs. A full corpus costs an estimated $13–37 with gpt-4o-mini; see [evidence/indexing_cost_estimate.json](audit/evidence/indexing_cost_estimate.json).

## Timeline (proposed)

| Dates | Milestone |
|---|---|
| 29 Sep – 2 Oct | Setup and Snellius smoke test. **Gate (2 Oct): a real MemGraphRAG run works.** |
| 3 – 6 Oct | Smoke test if not done; **schema-type histogram** (H3 go/no-go); SAM3 access check |
| 4 – 9 Oct | Data (E-VQA subset with MuKA KB images, MMQA dev subset), per-document loader, metrics, caption and dense baselines |
| 6 – 13 Oct | Visual layer: dispatch change, normalised seed channels, image nodes, type-guided SAM3 grounding, crop→entity seeds |
| 12 – 16 Oct | Main runs and ablations. **Results freeze 16 Oct.** |
| 5 – 22 Oct | Paper writing (background and method start early) |
| 23 Oct | Submission (9-page ACM) |

If the schedule slips, drop in this order: InfoSeek → MG²-RAG on MMQA → fine-grained-types arm → λ sweep. Keep the caption baselines and the H3 arms; they carry the claim. (The audit's §8 has an earlier, different plan; this table supersedes it.)

## Caveats

| Caveat | Status | Fallback |
|---|---|---|
| Integration only shown with stubbed LLMs/embedders | [V-run] mocked end-to-end | Real smoke test in week 1 |
| MemGraphRAG's schema types may be just OntoNotes NER labels | [V-read prompt], output unmeasured | Fine-grained `entity_type_extraction` prompt (+1 LLM call per chunk) |
| MemGraphRAG code lacks schema retrieval, type-node seeding (Eq. 7), IDF term (Eq. 8); type nodes get zero seed mass | [V-read, V-run]; also raised in upstream issue #6 | Call our baseline "MemGraphRAG (released code)"; implementing Eq. 7–8 is ~1 day if wanted |
| MemGraphRAG launch scripts crash; README paths wrong; `gritlm` is a hard import | [V-run] | Our own launch commands (see smoke-test task) |
| Default ontology filter dropped 2 correct facts; modified triples create duplicate entity nodes | [V-run, stub data] | Report as limitation; try `absolute` filter mode; select grounding candidates before the filter |
| SAM3 needs manual Hugging Face approval; built for generic noun phrases; licence requires acknowledging SAM in publications | Pending / [V-read] | Type→noun-phrase prompts. If no access by 6 Oct: Grounding DINO or OWLv2 (ungated) |
| EVA-CLIP-8B is a 33 GB download; 8.1B params → ≥16.2 GB VRAM in fp16 for weights alone | [V-read sizes, I VRAM] | SigLIP2-so400m (4.5 GB) for development; same encoder for us and MG²-RAG in the head-to-head |
| E-VQA (M2KR) is single-hop, recognition-dominated, one image per KB page | [V-run] | MMQA dev for H1–H3; official E-VQA two-hop items if needed |
| LLM budget: one MemGraphRAG index per corpus variant (KB, KB + captions, text, MMQA) ≈ $40–140 with gpt-4o-mini | [I], price [V-read] | Subsample; Batch API; self-host |
| Windows igraph cannot read GraphML | [V-run] | Run everything on Snellius (Linux) |

Every expected gain above is a hypothesis; nothing has been measured on real models yet.
