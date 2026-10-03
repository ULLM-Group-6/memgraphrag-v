# Methodology audit: proposed changes to MemGraphRAG-V (3 Oct 2026)

**Status:** proposal for group review (PR on branch `docs/methodology-audit`). Nothing here is decided until the group merges it.

**Inputs**

- [research-findings-2026-10-03.md](research-findings-2026-10-03.md): literature, datasets, models, prices, all checked against primary sources. Referred to below as *R§n* (section) or *R-Fn* (summary finding n).
- Code checks on the audit-time copies of both repos, done for this audit (§3). Referred to as *C1–C13*.
- What is being audited: [method.md](../method.md) and the [audit report](audit-report.md) as of 29 Sep, and the task guides.

Tags as elsewhere: **[V-run]** verified by running, **[V-read]** verified by reading code/paper, **[I]** inferred.

## 1. Verdict

The plan can still be built by 23 Oct, but not as written. Three problems would make the main experiments show nothing, and the novelty claim has to be narrowed.

1. **Blocker: on M2KR there are no knowledge-base images to index.** M2KR's E-VQA and InfoSeek passage corpora are text-only. The only image is the query image. "Image nodes linked to passages" and "SAM3 crops of KB images" have nothing to work on unless KB images come from somewhere else (R-F1).
2. **Blocker: the visual channel, as specified, would almost never act.**
   - Nearly every E-VQA/InfoSeek question is deictic ("this plant…"), so question text is useless for finding the entity (R-F4).
   - MemGraphRAG skips the graph entirely when no fact passes its similarity threshold (C2), which will be the common case for these questions.
   - The text seed vector is unnormalised and grows with corpus size, so λ has no stable meaning (C1).
3. **High: H3 may compare two near-identical label sets.**
   - MemGraphRAG's schema prompt suggests exactly the 18 OntoNotes NER labels that MG²-RAG's spaCy filter uses (R-F3).
   - H3 also confounds *where entities come from* (LLM triples vs spaCy NER) with *how they are filtered* (C5).
4. **High: the novelty margin is narrow.**
   - MG²-RAG (now ECCV 2026) already has image nodes, SAM3 crops linked to entities and visual PPR seeds.
   - HVM-GraphRAG (Jul 2026) already does conflict-resolving graph construction with images, via captions.
   - The only element no paper covers is choosing *which entities to ground* from LLM-derived types (R-F6, R§2.4).

None of this requires abandoning the architecture (Option A). It changes the data, the retrieval dispatch, the seed formula, the H3 design and the framing.

## 2. Findings and proposed changes

Severity: **blocker** (an experiment cannot work), **high** (a result or claim is likely invalid), **medium**, **low**. The column "This PR" says whether the change is already written into `method.md` / other docs in this PR, or left as a group decision (§5).

| # | Finding | Sev. | Evidence | Proposed change | This PR |
|---|---|---|---|---|---|
| A1 | M2KR KB passages are text-only (`EVQA_passages`: `language, passage_id, passage_content`; `Infoseek_passages`: `passage_id, passage_content, title`) | blocker | R-F1, R§3.1 [V-run parquet, re-checked via HF datasets-server API] | KB images from **MuKA's image-URL lists** (one Wikimedia image per M2KR passage/entity; covers all 51,472 E-VQA test passages). For section-level images use the **official E-VQA KB** (`image_urls` + `image_section_indices`). | method.md Data |
| A2 | M2KR's E-VQA test split has **no two-hop** items (1,000 templated + 2,750 automatic). Two-hop items (1,000) exist only in the official E-VQA `test.csv`. | blocker for the multi-hop story | R-F2, R§3.2 [V-run] | Either take two-hop questions from the official test set and KB (+≈1 day), or state that the KB-VQA set is single-hop and put multi-hop on a second dataset (A11). | D1 |
| A3 | The visual channel would rarely act on KB-VQA: questions are deictic (gold entity named in 34/3,750 E-VQA questions) and MemGraphRAG falls back to dense retrieval when no fact passes the threshold (`MemGraphRAG.py:1100-1105`). MG² Table 2: text→image R@1 is 1.8 (E-VQA) / 0.2 (InfoSeek), image→image 30.0 / 47.2. | blocker | R-F4, C2 [V-run, V-read] | (a) The visual query is the **query image** (question text only as a secondary visual query). (b) Visual seeds are computed **before** the fact gate. If no fact passes, run PPR on visual + passage seeds instead of falling back to dense retrieval. (c) Relax the `sum(node_weights) > 0` assertion (`:2175`). | method.md Retrieval |
| A4 | Text seeds are unnormalised: entity seeds are min–max fact scores divided by passage count; **every** passage gets 0.05 × its min–max DPR score (`:2124-2166`). Total passage mass grows with corpus size. `normalise(s_text + λ·s_visual)` therefore weights the visual channel differently per query and per corpus. | high | C1 [V-read] | L1-normalise each channel first: `s = (1−λ)·ŝ_text + λ·ŝ_visual`, `ŝ = s / ‖s‖₁` (MG²-RAG also weights branches, then normalises: `MMGraphRAG.py:985-1001`). Tune λ on a dev split. MG² Table 8 shows R@1 changes by only ~1 point over λ_v ∈ [0.5, 1.5], so don't expect a dramatic sweep. | method.md Retrieval |
| A5 | MemGraphRAG's schema prompt (`prompt.py:298-326`) lists the 18 OntoNotes labels as "common types"; "animal" is not among them. A ~200-type fine-grained prompt (`entity_type_extraction`, `prompt.py:2-295`) exists but is never called. H3 also confounds entity source with type filter. | high | R-F3, R§4.1, C5 [V-read] | (a) **Week-1 measurement:** histogram of schema types from the smoke test (task 01). (b) H3 as a four-arm comparison on the *same* LLM-extracted entities: no grounding / all entities / NER-label filter (MG²-style) / type-guided. Add a fine-grained-types arm (one extra LLM call per chunk) if the histogram is mostly OntoNotes labels. (c) Report grounding coverage and precision, not only retrieval. | method.md H3, ablations; task 01 |
| A6 | SAM3 is built for short generic noun phrases (32-token text encoder; "struggles to generalize to fine-grained out-of-domain concepts"). MG²-RAG prompts it with entity names such as "hinrich lichtenstein". How a type is judged "visual" is unspecified (types are free-form, per fact; one entity can have several, `MemGraphRAG.py:856-860`). | medium | R-F8, R§4.2, C6 | Map each *visual type* to a generic noun phrase ("animal", "plant", "building", "painting") and prompt SAM3 with that. Link each crop to the entity that owns the image's passage and has that type. Decide "visual" once per distinct type, by one LLM yes/no call or a reviewed list, and publish the list. | method.md Indexing |
| A7 | Visual PPR seeds (MG²-RAG, EviProp) and graph RAG with images (MG²-RAG, HVM-GraphRAG, RAG-Anything) already exist. | high | R-F6, R§2.4 | Narrow the claim: a controlled study of *which entities to ground* in a conflict-resolved LLM memory graph, and whether native visual evidence beats (entity-aware) captions under leakage controls. Cite MG²-RAG as ECCV 2026 and MemGraphRAG as KDD 2026. | method.md intro; README; D2 |
| A8 | The caption baseline as specified (KB images only, generic captions) is weak and unfair. Captioning the query image is what makes text retrieval work on KB-VQA, and gains can come from textual leakage. | high | R-F5, R§4.4 | Two caption baselines, both captioning **query and KB images** with the same MLLM used for answering: generic, and entity-aware ("name the most specific entity, then describe"). Captions are indexed as separate passages linked to their document (keeps LLM-cache hits, A14). Report results with and without query-side captions. Decide whether Wikipedia image captions may be used (they leak names). | method.md baselines |
| A9 | H1 against text-only MemGraphRAG is a foregone conclusion on KB-VQA: a text-only system cannot know which plant "this plant" is. | high | R§5.2 | Keep H1 vs text-only as a sanity check. The real comparisons are against **dense CLIP image→image/text retrieval**, **MG²-RAG**, **zero-shot MLLM** (no retrieval) and the caption baselines. | method.md Evaluation |
| A10 | Adding image (and crop) nodes changes the random walk even at λ = 0: passages with images gain degree and leak mass to image nodes. | medium | C3 [I, from `run_ppr` on the shared graph] | H4 compares the full system on text-only questions with the **unmodified** MemGraphRAG graph. Rename the λ = 0 ablation "visual graph, no visual seeds". | method.md Evaluation |
| A11 | On KB-VQA each KB document has one image of its main entity ("visual shortcut", RETINA), so crops add little and memory/multi-hop has no room. H1–H3 need a corpus where images live on the corpus side, show several entities, and questions are text. | high | R§3.1, R§3.3, R§5.3 | Add **MMQA (MultimodalQA) dev**: 2,441 questions, 940 need images, 569 compositional + image; 57k Wikipedia entity images (2.36 GB); gold doc IDs. Check its licence first (not stated). | D1 |
| A12 | SigLIP2 truncates text at **64** tokens (not 77). Swapping EVA-CLIP-8B for SigLIP2 is an uncontrolled change relative to MG²-RAG, which has no encoder ablation. | medium | R-F9, R§4.3 | Use the same encoder in MG²-RAG and ours for the head-to-head, or report that comparison with EVA-CLIP-8B. Optionally try `Qwen3-VL-Embedding-2B` (Apache-2.0, 32k context) for joint image+question queries. | method.md Design choice 1; D3 |
| A13 | `code/index.py` concatenates the corpus and slides 256-token windows across document boundaries, prefixing `idx:` (`index.py:16-37, 78-80`). Chunks have no document ID, so images cannot be linked to passages and gold IDs are lost. But titled per-passage corpora ship with the repo (`dataset/*/*_corpus.json`; 9,811 HotpotQA passages). | medium | C4 [V-run] | Our loader passes **one document (or section) per chunk** straight to `index_with_memory`, bypassing `index.py`'s splitter. Gold IDs come from `supporting_facts` titles (text) or `pos_item_ids` (M2KR). | method.md Pipeline |
| A14 | Only the text-regression corpus has a cost estimate. The multimodal KB (2–5k documents) costs about as much as HotpotQA ($13–35 with gpt-4o-mini). The caption baselines need at least one more full index, and every chunk whose text changes misses the LLM cache. | medium | C12; R§4.6 (price still $0.15/$0.60 per 1M) | Captions as separate passages (unchanged chunks hit the SQLite cache). Budget one MemGraphRAG index per corpus variant: multimodal KB, KB + captions, text regression, and MMQA if adopted. At $13–35 each (HotpotQA-sized), that is ≈ $40–140, above P10's < $50 unless corpora are subsampled, the Batch API (half price) is wrapped in, or the LLM is self-hosted. Re-estimate from smoke-test token counts. Avoid `gpt-4.1-nano` (shutdown 23 Oct 2026). | method.md Caveats |
| A15 | The default `percentile` ontology filter deletes facts, and with them entities, before any grounding happens. | low–medium | C7, audit §2.1 | Select grounding candidates from the schema-annotated memory *before* the ontology filter, or run with `--ontology-filter-mode absolute`. Report how many groundable entities the filter removed. | method.md Indexing |
| A16 | The plan says "nearest image and crop nodes become seeds" and Option A says "optionally region" nodes. MG²-RAG keeps crops out of the graph and maps crop similarity onto the linked entities (`MMGraphRAG.py:943-1001`). The plan also omits MG²'s query-image → text paths (image→sentence/chunk), which on a text-only KB are its main visual signal. | medium | C8, C9 [V-read] | Image nodes in the graph; crops **not** nodes: a crop match seeds its linked entity (MG² style). Only one new layer (`image`) to add to the `:1905` assertion. Image→passage-text matching in CLIP space is an ablation, not the default (64/77-token text limit). | method.md Pipeline |
| A17 | Our numbers cannot be compared with either paper (different KB sizes, splits, encoders, metrics). The released MemGraphRAG retrieval is HippoRAG 2's seed rule on a different graph. | medium | R-F10, R§3.4, R§4.8 | Never put subset numbers next to published ones; re-run every baseline. Names: "MemGraphRAG (released code)", and the legacy `index()` path as "w/o memory (≈ HippoRAG 2 pipeline)". | method.md Evaluation |
| A18 | Metrics: E-VQA's official BEM needs TensorFlow + TF Hub. M2KR's "InfoSeek test" is actually InfoSeek **validation** (test answers unreleased). | low | R§3.2, R§5.5 | Containment + LLM judge for all sets, and BEM if TF installs on Snellius. Call InfoSeek "InfoSeek val (M2KR test split)". | method.md Evaluation |
| A19 | Good news: query images are available **ungated** in `BByrneLab/M2KR_Images` (E-VQA iNat 8.92 GB zip, readable per file with range requests; GLDv2 2.79 GB). OVEN and AToMiC are `gated=auto`, not manual. | medium (positive) | R-F7 [V-run] | Replace "fetch only the needed images (OVEN/AToMiC out of scope)" with this source. | method.md Data; audit errata |
| A20 | SAM License allows use and redistribution with the licence attached, and **requires acknowledging SAM in publications**. MG²-RAG's vendored `sam3/` ships without that licence. | low | R§4.2 | Load SAM3 via transformers (`Sam3Model`) instead of copying the vendored package. Acknowledge SAM 3 in the paper. | method.md Caveats |

## 3. Code-level checks (C1–C13)

All on the audit-time copies (`work/MemGraphRAG`, `work/MG2-RAG`). Line numbers will drift; re-locate them in your clone.

- **C1 Seed scale.**
  - `graph_search_with_fact_entities` builds `node_weights = phrase_weights + passage_weights` and passes it to igraph without normalising (`MemGraphRAG.py:2169, 2178`).
  - Phrase weight is the last matching fact's min–max score divided by the entity's passage count (`:2124-2141`).
  - Every passage gets `min_max(DPR) × 0.05` (`:2157-2164`).
  - [V-read]
- **C2 Dispatch.** `retrieve_single_query` calls `dense_passage_retrieval` when `top_k_facts` is empty (`:1100-1105`). The CLI default for the raw-cosine threshold is 0.6 (`retrieval_dataset_test.py:248`). [V-read]
- **C3 Topology.** PPR runs on one shared graph (`run_ppr`, `:2299-2333`). Adding nodes and edges changes every query's walk, whatever the seeds. [V-read; size of the effect I]
- **C4 Chunking.** `split_text` encodes the *whole* corpus string and emits 256-token windows with 32 overlap. `main` prefixes `f"{idx}:"` (`index.py:16-37, 78-80`). The `*_corpus.json` files hold `{idx, title, text}` per passage (HotpotQA 9,811; 2Wiki and MuSiQue too). [V-run]
- **C5 H3 confound.** `get_filtered_entities` keeps only spaCy entities labelled FAC, LOC, ORG, PERSON, PRODUCT or WORK_OF_ART (`MMGraphRAG.py:577-585`). spaCy NER spans rarely include common nouns such as "impala", while MemGraphRAG's LLM triples do (`MemGraphRAG.py:832-845`). [V-read]
- **C6 Types per fact.** `build_memory_graph` adds an entity to the type set of every schema it appears in (`:856-860`), so an entity can have several types. [V-read]
- **C7 Filter before grounding.** `index_with_memory` runs schema extraction → ontology filter → conflict detection and resolution before graph construction (`:992-1026`). `initial_memory_with_schema.json` is always written (`:994-996`), so pre-filter types are available. [V-read]
- **C8 Crops in MG²-RAG.** Crop→entity links live in `subimg_to_entity_map`; text/image→crop matches add seed weight to the linked entities (`MMGraphRAG.py:943-1001`). [V-read]
- **C9 Missing paths.** MG²-RAG's image branch also matches the query image against sentences and chunks (`i2s`, `i2chunk`, `:953-975`). The plan's visual channel lists only image and crop nodes. [V-read]
- **C10 ID merging (P1) is less load-bearing than the docs say.** Option A never imports MG²-RAG's graph: we compute grounding edges against MemGraphRAG's own node IDs. Hash equality across repos matters only if MG²-RAG artefacts are reused. Inside MemGraphRAG, query-time entity lookup hashes `f[0].lower()` (`:2125-2135`), not `text_processing`. Normalisation consistency is therefore an *internal* MemGraphRAG question (it is the modified-triple bug, audit §2.1). [V-read]
- **C11 One image per document.** MG²-RAG zips one image ID per document (`MMGraphRAG.py:599-603`). Wikipedia sections and MMQA have 0..n images per document, so our loader must support that. [V-read]
- **C12 Cost.** See A14. [I]
- **C13 Doc drift.**
  - `method.md` dropped the zero-shot-MLLM baseline that the audit listed (cheap, and needed to show retrieval helps at all).
  - README's status line is dated 29 Sep.
  - The 2 Oct smoke-test gate has passed, and its outcome is not recorded in the repo.

## 4. Corrections to existing docs

| Where | Claim | Correct | Source | Fixed in this PR |
|---|---|---|---|---|
| `method.md` Design choice 1 | "CLIP text encoders also truncate at 77 tokens" | SigLIP2-so400m (the development encoder) truncates at **64**; EVA-CLIP-8B at 77 | R§4.3 | yes |
| `method.md` Data, audit §1.4, §7 | InfoSeek/E-VQA images need OVEN 294 GB / AToMiC 174 GB, "gated, out of scope" | Query images are ungated in `BByrneLab/M2KR_Images` (≈12 GB for E-VQA; 9 GB tar for InfoSeek val). OVEN/AToMiC are `gated=auto`. KB images are *not* in M2KR at all. | R-F1, R-F7 | yes (method.md); errata note in audit |
| `method.md` Caveats; audit §6, §7 | "Prefer E-VQA two-hop items" | Not possible from M2KR: two-hop items exist only in the official E-VQA `test.csv` | R-F2 | yes |
| README idea; `method.md` Design choice 2 | Schema types are things like "animals, buildings" | The schema prompt suggests the 18 OntoNotes NER labels (no "animal"); actual output is unmeasured | R§4.1 | yes |
| README, `method.md` upstream tables | MG²-RAG cited as arXiv only | ECCV 2026; MemGraphRAG is KDD 2026 | R§2.3 | yes |
| audit §2.1 Datasets row, §6 | Gold passage boundaries are lost | Lost only through `index.py`'s splitter; per-passage `*_corpus.json` files with titles ship with the repo | C4, R§5.6 | errata note |
| audit §3.2, `method.md` Caveats | EVA-CLIP-8B ≈16 GB VRAM fp16 | 8.1B params → ≈16.2 GB for weights alone; plan ≥18–20 GB [I] | R§4.3 | yes |
| audit §2.1 NV-Embed/BGE row | BGE model card recommends CLS + instruction [I] | Confirmed: upgrade to [V-read]; `encode()` also silently drops the instruction kwarg | R§4.7 | errata note |
| `method.md` Data | "M2KR InfoSeek/E-VQA test splits" | M2KR's InfoSeek "test" is InfoSeek **validation** | R§3.1 | yes |

## 5. Decisions for the group

These change scope or framing, so they are left open. Each has a recommendation.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Evaluation data | (a) M2KR E-VQA subset + MuKA KB images only; (b) (a) + **MMQA dev** image questions; (c) (a) + official E-VQA two-hop items with the official KB | **(b).** KB-VQA alone can only show "image→image retrieval helps", which MG²-RAG already shows. MMQA is where grounding and multi-hop can matter. Do (c) only if the memory/multi-hop story must sit on E-VQA. Drop InfoSeek unless time allows (no extra hop structure, 9 GB tar). |
| D2 | Contribution claim | (a) "MemGraphRAG with native images" (current); (b) controlled study of grounding-selection policies and native vs caption evidence on a memory graph | **(b).** (a) is largely covered by MG²-RAG and HVM-GraphRAG. |
| D3 | Visual encoder | SigLIP2 throughout; EVA-CLIP-8B for the final and MG² runs; Qwen3-VL-Embedding-2B | SigLIP2 for development. For the head-to-head with MG²-RAG, run **both systems with the same encoder**. |
| D4 | MLLM for captions and answers | Qwen2.5-VL-7B (matches MG² Table 3); Qwen3-VL-8B / 4B (newer, Apache-2.0) | Qwen3-VL-8B (4B if VRAM-bound), the **same model for every system**. |
| D5 | Implement MemGraphRAG's missing Eq. 7–8 | yes (~1 day) / no | **No** (unchanged). Name the system "MemGraphRAG (released code)". |
| D6 | BGE wrapper deviations (mean pooling, dropped instruction) | keep as released / fix everywhere | Keep as released and state it. |
| D7 | Go/no-go on H3 | — | If the week-1 histogram shows almost only OntoNotes labels, the "type-guided" arm uses the fine-grained `entity_type_extraction` prompt (+1 LLM call per chunk). |
| D8 | SAM3 fallback | wait / switch | If access isn't granted by **6 Oct**, use Grounding DINO or OWLv2 (ungated, in transformers) with the same type→noun-phrase prompts. |

## 6. Proposed timeline (supersedes `method.md` table once merged)

| Dates | Milestone |
|---|---|
| 3–6 Oct | Smoke test if not done; **schema-type histogram** (H3 go/no-go, D7); SAM3 access check (D8) |
| 4–8 Oct | Data: E-VQA subset builder (M2KR text + MuKA KB images + `M2KR_Images` query images); MMQA dev subset if D1(b); per-document loader (A13); metrics |
| 5–9 Oct | Caption baselines, generic and entity-aware (A8); dense CLIP and zero-shot MLLM baselines |
| 6–13 Oct | Visual layer: dispatch change (A3), normalised channels (A4), image nodes, type→noun-phrase SAM3 grounding (A6), crop→entity seeds (A16) |
| 12–16 Oct | Main runs and ablations; λ on dev split. **Results freeze 16 Oct.** |
| 5–22 Oct | Writing |
| 23 Oct | Submission |

If the schedule slips, drop in this order: InfoSeek → MG²-RAG baseline on MMQA → fine-grained-types arm → λ sweep. Keep the caption baselines and the H3 arms; they carry the claim.
