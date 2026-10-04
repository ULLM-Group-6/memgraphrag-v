# Novelty re-check for decision D2 (4 Oct 2026)

Scope: checks the D2 contribution claim (`docs/audit/methodology-audit-2026-10-03.md`) against the method sections of five close papers, new arXiv work since 1 Sep 2026, and forward citations of MG²-RAG and MemGraphRAG. It builds on `docs/audit/research-findings-2026-10-03.md` §2 and does not repeat its checks.

Tags: **[V-read]** read in the primary source at the cited section. **[V-run]** produced by running a query (arXiv API, Semantic Scholar API, OpenAlex API, Google Scholar HTML). **[I]** inferred.

Method: arXiv HTML for each paper was downloaded with curl, converted to text locally, and the method sections were read in full. This replaces the WebFetch summaries that §6 of the 3 Oct file flagged.

**D2 claim under test:** "a controlled study of WHICH ENTITIES TO GROUND when building a multimodal graph for RAG. Using a conflict-resolved LLM memory graph, compare prompt-selection policies for concept segmentation (no grounding / all entities / NER-label filter as in MG²-RAG / LLM-schema-type-guided mapped to SAM3 noun phrases), measuring grounding coverage/precision and retrieval; and native visual PPR seeds vs (entity-aware) captions under leakage controls."

---

## 1. Method sections of the five close papers (Q1)

Each paper is checked on three points:
- **(a) Type-based selection:** does it choose which entities/objects to ground or segment by entity type, ontology or schema?
- **(b) Text-prompted localisation:** does it use detection or segmentation prompted by text?
- **(c) Native vs caption:** does it compare native visual evidence with captions?

### HVM-GraphRAG ([arXiv 2607.24861](https://arxiv.org/html/2607.24861), Jul 2026)

**What it does [V-read].** It parses documents into chunks with a layout parser and organises them into a document tree (§3.1.1). Each chunk goes to a modality-specific extractor that returns concepts, entities, schemas and facts (§3.1.2, Eq. 4). Facts are checked against a "Cross-Modal Holistic View" by grouping on the keys (h,r), (h,t) and h. An LLM detector and resolver then process each group (Eqs. 5–11). This is MemGraphRAG-style conflict resolution, and the paper cites MemGraphRAG (ref. list line "MemGraphRAG: memory-based multi-agent system…").

**How images are handled [V-read].** Qwen2.5-VL is the "visual-language extractor" (App. B.4). The image prompt (App. D.1, Fig. 15) tells the VLM to make the whole image an `IMAGE` entity and to identify "distinct physical objects, people, animals… labels, titles". Each `entity_type` "MUST be one of the following types: {entity_types}". So visual entities are **type-constrained by a predefined list**, but the VLM names them in text: there is no box, mask or crop.

**Retrieval [V-read].** Image chunks are turned into VLM text summaries, both for query–chunk similarity (§3.2.3, Eq. 21: "For image chunks, it uses a VLM to generate textual summaries") and for answering. The ablation (§4.4) removes the concept graph, conflict resolution and modality-aware organisation. None of the ablations swaps captions for native visual evidence.

**Answers:**
- (a) Partly. An entity-type list constrains what the VLM extracts from images, but nothing is localised.
- (b) No.
- (c) No. It is caption-only in retrieval. Case B.8.2 only argues informally that it beats caption-based GraphRAG.

**Verdict: partial threat on wording only.** It is the nearest prior work for "conflict-resolved graph + images + entity types". It blocks any wording such as "first to use entity types to decide what is extracted from images". It does **not** do localised grounding (segmentation/detection), compare selection policies, use visual seeds, or compare native vs caption evidence.

### mKG-RAG ([arXiv 2508.05318](https://arxiv.org/html/2508.05318), SIGIR 2026, DOI 10.1145/3805712.3809680)

**What it does [V-read].**
- Text graph: an MLLM prompt extracts entities and relations (§3.1.1).
- Visual graph: the scene-graph model EGTR returns "visual objects with predicted category labels and bounding boxes" (§3.1.2; EGTR named in §4.1 Implementation Details). This is a closed-vocabulary detector. It is **not** prompted with text and **not** filtered by entity type.
- Alignment: an MLLM matches the whole object list to the text entities after detection (§3.1.3, Eq. 1, Fig. 5 prompt). The matched region becomes an attribute of the text node.
- Retrieval: dense retrieval with a trained QM-Retriever (BLIP-2 Q-Former) over documents, then over entities/relations, then BFS expansion (§3.2). There is no PPR.

**Caption comparison [V-read].** §4.2.1 / Table 1 report a "text-only" variant that "uses both questions and image captions as queries" with CLIP-text, against a vision-only CLIP variant; the text-only variant does better. §4.3.2 replaces graph retrieval with chunks chosen by "question and image caption". It also argues that RAG-Anything loses modality information by converting images to text (§4.2.1).

**Answers:**
- (a) No.
- (b) No. Detection is closed-vocabulary.
- (c) Partly. There is a query-side caption-vs-vision retriever variant on E-VQA/InfoSeek, but no leakage control and nothing on the corpus or graph side.

**Verdict: not a threat to the selection half. It weakens any "first native-vs-caption comparison" wording.** It is a direct KB-VQA competitor to cite and discuss. Its Table 1 finding (captions plus question beat vision-only CLIP) supports making our caption baseline strong.

### RAG-Anything ([arXiv 2510.12323](https://arxiv.org/html/2510.12323))

**What it does [V-read].** Parsers split documents into atomic units (§2.2). For each non-text unit an MLLM produces a detailed description and an entity summary ("entity name, type, and description"). Entities and relations are extracted from that **description** (Eq. 2) and attached to a multimodal anchor node through `belongs_to` edges (Eqs. 3–4). The text KG follows LightRAG. The two graphs are merged by entity name (§2.2.2).

**Retrieval [V-read].** Retrieval uses a text embedding of the query, keyword/entity matching, neighbourhood expansion and dense matching (§2.3). The original images are retrieved by reference only for the answering VLM (§2.4(ii): "Textual proxies enable efficient retrieval while authentic visual content provides rich semantics… during synthesis"). The ablations (§3.3, Table 4) are chunk-only and without the reranker.

**Answers:**
- (a) No.
- (b) No. There is no detection or segmentation.
- (c) No. Retrieval uses captions only, and the paper reports no native-vs-caption test.

**Verdict: not a threat.** It is the natural off-the-shelf "caption-centric multimodal graph RAG" baseline or reference, as already proposed.

### MMGraphRAG ([arXiv 2507.20804](https://arxiv.org/html/2507.20804), v3 18 Jul 2026; venue still not stated in the paper)

**What it does [V-read].** The paper states that the graph "does not rely on RDF typing, formal ontologies, or entailment rules" (§3). Its text side uses GraphRAG extraction with "a lightweight default type set". Img2Graph (§3.2, App. E) works in five steps:
1. YOLOv8 segmentation produces "image feature blocks". This is class-based instance segmentation, not text-prompted.
2. An MLLM describes each block (object / organism / person).
3. Entities and relations are extracted from those descriptions.
4. Regions are aligned to entities.
5. A global image entity is added.

Cross-modal linking uses SpecLink, which does spectral clustering of nearby text entities followed by an LLM choice (§3.3.1). Retrieval is text-embedding top-k seed entities plus one-hop expansion (§3.1).

**Caption comparison [V-read].** §4.2 states that prior multimodal GraphRAG methods "effectively reduce to text-based GraphRAG with captions replacing images". Its "LLM" baseline converts images to text with an MLLM. Its own visual nodes, however, are still MLLM descriptions of regions, so retrieval never uses native visual features.

**Answers:**
- (a) No. Segmentation covers every YOLO region, with no type-based choice.
- (b) No. It uses a class-based segmenter.
- (c) Only at system level (with images vs images converted to text), with no leakage control.

**Verdict: not a threat.** Cite it as region-level image graph prior work.

### CEMMKG ([arXiv 2608.25986](https://arxiv.org/html/2608.25986), Aug 2026)

**What it does [V-read].** It keeps MMGraphRAG's Img2Graph backbone unchanged ("We leave the backbone f_v unchanged and replace only its context argument", §4.2.1, Eq. 6). It builds local context (surrounding text, reference sentence, paragraph or summary) and global context (abstract or document summary) for each figure (§4.1). This context goes into the MLLM description step and the SpecLink fusion step (§4.2.2, Eq. 7). Retrieval adds a rule that prefers figures or tables named in the query, with a BM25 fallback (§5.1.4). It is evaluated on a 106-question "VisionHeavy" subset of MMLongBench-Doc.

**Answers:**
- (a) No.
- (b) No. YOLO is inherited, and the text conditions the description, not the localisation.
- (c) No.

**Verdict: not a threat.** It shares only the idea that text should guide visual extraction. It cites MG²-RAG (ref. list, `External Links: 2604.04969`).

### Q1 summary

None of the five chooses *which graph entities to localise* by type, schema or NER label. None prompts a detector or segmenter with text derived from entity types. None compares selection policies.
- The only text-prompted segmentation in the set remains MG²-RAG itself: SAM3 prompted with names of spaCy entities from 6 NER labels (3 Oct file, finding 3/8).
- HVM-GraphRAG constrains VLM entity *extraction* with an allowed type list, without localisation. This is the one point where the wording must be careful.
- Native-vs-caption comparisons exist in weaker forms: mKG-RAG Table 1 (query side, no leakage control), MMGraphRAG's "LLM" baseline (system level), and §2 below.

---

## 2. New papers since 1 Sep 2026 (Q2)

### Searches run [V-run]

- **arXiv API**, 30 queries restricted to `submittedDate:[20260901 TO 20261004]`, covering:
  - multimodal + graph + RAG + segmentation/grounding/SAM/detection;
  - "multimodal knowledge graph" + retrieval;
  - GraphRAG/graph RAG + image;
  - PPR + image;
  - entity type/ontology/schema/taxonomy + grounding/segmentation + RAG;
  - SAM3/"Segment Anything" + RAG/KG;
  - Grounding DINO/open-vocabulary + retrieval/KG;
  - KB-VQA/E-VQA/InfoSeek + graph;
  - HippoRAG + multimodal;
  - memory graph/conflict + multimodal;
  - caption + leakage;
  - "promptable concept";
  - names of MG²-RAG, MemGraphRAG, mKG-RAG, MMGraphRAG, RAG-Anything;
  - "type-guided/ontology-guided/schema-guided" + visual;
  - "visual grounding" + KG construction;
  - "which entities/entity selection/prompt selection" + segmentation;
  - caption + RAG + image;
  - open-vocabulary + RAG.
- **Google Scholar**, 5 date-sorted queries (`as_ylo=2026`).
- **Web search**, 5 queries.

Some arXiv API phrase queries return surprisingly few hits, so absence is not proof.

### Results

| Paper | Date | What it does | Verdict |
|---|---|---|---|
| *Signal or Noise? Modality Contribution and Cooperation in Multimodal GraphRAG* ([2609.35304](https://arxiv.org/abs/2609.35304), Univ. of Amsterdam) | 28 Sep 2026 | Adds modality provenance to RAG-Anything's graph edges and limits retrieval to subsets of {text, table, image, layout}. Measures contribution and cooperation (SHAPE) on two DocVQA benchmarks with 5 MLLMs. Finds that text and tables contribute most and that combining modalities is often redundant. §3–§5.1 [V-read]. | **Cite, not a threat.** It ablates modalities at inference time; it does not select what to ground, segment or compare native vs caption representations. Its "redundancy, not synergy" result supports our leakage-control framing. |
| *PILAR* ([2609.32895](https://arxiv.org/abs/2609.32895)) | 26 Sep 2026 | Entity-linked assertion graph over text, table and figure assertions for multimodal document ODQA. App. B.1 compares description-based with structured visual assertions. Visual assertions hurt alone and help only after locality filtering (abstract, §4.4) [V-read]. | Cite. Document QA with no object grounding. |
| *TAEC* ([2609.37349](https://arxiv.org/abs/2609.37349)) | 29 Sep 2026 | Training-free multi-step visual RAG over pages; cites MG²-RAG [V-read refs]. | Not relevant. |
| *TrioRAG / Less Is More* ([2609.19417](https://arxiv.org/abs/2609.19417)) | 16 Sep 2026 | Already in the 3 Oct table. | Unchanged. |
| *GraMRAG* ([2609.14066](https://arxiv.org/abs/2609.14066)) | 12 Sep 2026 | Already in the 3 Oct table. | Unchanged. |
| *PRISM-RAG / NicoPRISM* ([2609.23769](https://arxiv.org/abs/2609.23769)) | 20 Sep 2026 | Multimodal hypergraph RAG over product images, attribute captions and legislation [V-read abstract]. | Not relevant. It is caption-based and domain-specific. |
| *ViTeGate* ([2609.14685](https://arxiv.org/abs/2609.14685)) | 13 Sep 2026 | Knowledge poisoning for vision-language RAG. | Not relevant. |
| *Grounded Entity Biographies* ([2609.38155](https://arxiv.org/abs/2609.38155)) | 29 Sep 2026 | Long-video memory that groups visually grounded instances of one entity [V-read abstract]. | Not relevant (video, instance tracking). |

### Missed by the 3 Oct table (before 1 Sep, found today)

- **Comparison of Text-Based and Image-Based Retrieval in Multimodal RAG** ([2511.16654](https://arxiv.org/abs/2511.16654), Nov 2025) [V-read abstract]. A controlled comparison of LLM-summary (caption) retrieval vs native multimodal-embedding retrieval: +13 pt mAP@5 for native. It is flat RAG on a 40-question financial benchmark. **This is prior art for "native vs caption", so the D2 claim must not present that comparison as new in itself.** What stays specific to us: graph RAG with PPR seeds, entity-aware captions as the competitor, and leakage controls.
- **KBMR, Beyond Visual Similarity: Entity-Aligned Retrieval for KB-VQA** ([2608.21450](https://arxiv.org/abs/2608.21450), ACM MM 2026) [V-read abstract]. An MLLM-based embedding retriever for KB-VQA that preserves entity identity; +14.7 R@1 over CLIP. Cite it, and possibly compare in the E-VQA/InfoSeek setting. Not a graph method and not a threat.
- **HANIA** ([2608.29088](https://arxiv.org/abs/2608.29088), ISWC 2026 GLOW workshop) [V-read abstract]. A VLM extracts question-relevant visual statements, followed by a per-input multimodal graph and pruning, evaluated on ScienceQA. Not a threat.

### Q2 verdict

No new paper since 1 Sep 2026 (up to 4 Oct, arXiv API) studies entity-type, ontology or schema-guided choice of what to ground or segment for graph RAG. None compares grounding-selection policies. **No new threat.**

---

## 3. Forward citations (Q3)

### Semantic Scholar Graph API, 4 Oct 2026 [V-run]

No change since 3 Oct:
- MG²-RAG: 2 citations (ZipRerank 2605.11864; Position 2506.08354).
- MemGraphRAG: 1 citation (2607.22319).

### Google Scholar, reachable today [V-run]

**MemGraphRAG** (`cites=2262286363289275481`, "Cited by 4"):
1. HVM-GraphRAG (2607.24861);
2. *Graph Engineering in the Era of LLM Agents* ([2608.21156](https://arxiv.org/abs/2608.21156), 21 Aug 2026, survey sharing authors with MemGraphRAG);
3. *MemSyco-Bench* ([2607.01071](https://arxiv.org/abs/2607.01071), benchmark of sycophancy in agent memory, same group);
4. 2607.22319.

Scholar's search page also lists a Japanese JSAI SIG paper on how KG quality affects graph-based RAG retrieval ([J-STAGE](https://www.jstage.jst.go.jp/article/jsaisigtwo/2026/SWO-069/2026_06/_article/-char/ja/)); it matched the "MemGraphRAG" query but was not in the cited-by list.

**MG²-RAG:** Scholar now lists the ECCV 2026 proceedings version: Springer LNCS, DOI [10.1007/978-3-032-37167-6_32](https://link.springer.com/chapter/10.1007/978-3-032-37167-6_32). OpenAlex confirms the DOI. **Use this DOI in the bibliography.** The arXiv record (`cites=10308882694258077805`) is cited by HVM-GraphRAG and CEMMKG. The Springer record shows the 2506.08354 position paper. A plain "MG2-RAG" search also returns TAEC (2609.37349), which cites it [V-read refs].

**OpenAlex** full-text search [V-run] confirms that HVM-GraphRAG mentions both base papers and CEMMKG mentions MG²-RAG.

### Q3 verdict

Nothing published after 3 Oct. The citers missing from Semantic Scholar are HVM-GraphRAG, CEMMKG, TAEC, the Graph Engineering survey and MemSyco-Bench. All except TAEC were known or are irrelevant; TAEC is irrelevant. **No new threat.**

**One thing to note [I]:** HVM-GraphRAG cites both MemGraphRAG and MG²-RAG and combines conflict resolution with images. Reviewers will see it as the nearest neighbour, so the related-work section should name it and state the difference: caption-only, no localisation, no selection study.

---

## 4. Wording implications for D2 [I]

1. **Keep "which entities to ground", and say the grounding is localised.** Write "localised grounding (text-prompted concept segmentation masks/crops)". HVM-GraphRAG already uses a type list to constrain *VLM entity extraction from images* (App. D.1, Fig. 15), so "type-guided visual extraction" by itself is not new. Type-guided choice of *what to segment* is new.
2. **Frame native-vs-caption as a secondary, controlled analysis, not a novelty.** It extends [2511.16654](https://arxiv.org/abs/2511.16654) (flat RAG), [2607.16604](https://arxiv.org/abs/2607.16604) (leakage) and mKG-RAG Table 1 (query-side captions vs vision) to graph RAG with PPR seeds, using entity-aware captions on both the query and the corpus side.
3. **Name HVM-GraphRAG in the claim's contrast sentence**, alongside MG²-RAG. MG²-RAG has localised grounding with an NER-label filter and no conflict resolution. HVM-GraphRAG has conflict resolution and type-constrained VLM extraction, but no localisation and is caption-only.
4. **Cite MG²-RAG by its ECCV/LNCS DOI** 10.1007/978-3-032-37167-6_32.
