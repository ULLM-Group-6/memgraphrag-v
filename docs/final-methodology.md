# MemGraphRAG-V: final methodology (4 Oct 2026)

**Status:** final proposal for ratification at the 4 Oct group call. Once ratified it is the spec we build and write against. Where it differs from [method.md](method.md) or the [3 Oct methodology audit](audit/methodology-audit-2026-10-03.md), this document wins. Its timeline replaces both earlier tables.

**How it was made:**
- The 3 Oct audit's recommendations for decisions D1–D8 are adopted as defaults.
- Three checks were added on 4 Oct, with full evidence in `docs/audit/`:
  - [consistency check](audit/consistency-check-2026-10-04.md): docs vs each other and vs code;
  - [novelty re-check](audit/novelty-recheck-2026-10-04.md): method sections of the five closest papers, plus papers since 1 Sep;
  - [data licences and releases](audit/data-licences-2026-10-04.md).
- Tags as elsewhere: **[V-run]**, **[V-read]**, **[I]**.

**Status going into the call (updated 4 Oct):**
- **G1 passed.** MemGraphRAG runs end to end on Snellius with Qwen2.5-7B (≈ 16 GB). With the team's patch and sampling override, every stage completed: OpenIE, schema extraction, memory graph, retrieval and QA.
- **G3 passed.** SAM3 access has been granted on Hugging Face.
- **Everything runs on Snellius.** The only possible external cost is an optional gpt-4o-mini judge (≈ $1–3); see §6.4.

Section 7 lists what the call must confirm.

---

## 1. What we claim

> We study **which entities to localise** when building a multimodal graph for RAG. On a conflict-resolved LLM memory graph (MemGraphRAG), we compare selection policies for text-prompted concept segmentation (SAM3):
> - no grounding;
> - all entities;
> - an NER-label filter (as in MG²-RAG);
> - LLM schema types mapped to noun phrases.
>
> We measure grounding precision and coverage, and their effect on retrieval. As a secondary controlled analysis, we compare native visual PPR seeds with generic and entity-aware captions, under leakage controls.

In the contrast sentence, name both nearest neighbours:
- **MG²-RAG** localises entities but selects them by NER label.
- **HVM-GraphRAG** constrains what a VLM extracts from images with an entity-type list, but localises nothing and works only through captions.

Native-vs-caption is not a new idea: 2511.16654, 2607.16604 and mKG-RAG's Table 1 compare it in weaker forms. Present it as a controlled analysis, not as a contribution. [V-read, novelty re-check §1–2]

---

## 2. What changed

### 2a. New on 4 Oct (not in any earlier doc)

| # | Area | Before (method.md / audit, 3 Oct) | Now | Why |
|---|---|---|---|---|
| 1 | **H3 outcome** | "Type-guided grounding selects better entities, *and that improves retrieval*" | **Primary: grounding precision and coverage.** Retrieval across arms is secondary and reported whatever it shows. | Both datasets have one image of one main entity per page: 99.3 % of MMQA entities have exactly one "profile image", and MuKA gives one image per E-VQA page [V-run]. A crop of the only depicted entity adds little over the whole image, so a retrieval effect is unlikely to be detectable. |
| 2 | **H3 arms** | Arms differ in selection *and* SAM3 prompt form (entity names vs type noun phrases). NER arm on LLM entities undefined. | Every arm prompts SAM3 with **type → noun phrase**; arms differ **only in which entities are selected**. The NER arm is defined via spaCy span matching (§4.4). | Removes the confound between *which* entities are selected and *how* SAM3 is prompted. |
| 3 | **Crop → entity link** | "link each crop to the matching entity" (ambiguous) | Link to the page's title entity if it matches the noun phrase. Otherwise link to every matching candidate with weight 1/k. | Several entities can share one noun phrase ("plant"). |
| 4 | **Graph topology** | Unspecified whether grounding adds edges | Grounding adds **no edges**: crops only seed entities. Image nodes, and image–passage edges, are the same in every arm. | Keeps the graph identical across H3 arms, so one λ = 0 run serves all of them. MG²-RAG *does* add image→entity edges (`MMGraphRAG.py:612-616`); say that we differ. |
| 5 | **Grounding candidates** | "Take candidates from the schema memory *before* the ontology filter" | Candidates are entities that **have a node in the final graph**. Their types are read from the pre-filter `initial_memory_with_schema.json`. Report how many visual-typed entities the filter removed. | Entity nodes are created only from facts that survive filtering (`MemGraphRAG.py:831-841`), so pre-filter entities may have no node to seed [V-read]. |
| 6 | **KB unit** | "Gold + distractor pages", passage IDs and MuKA keys used interchangeably | Chunk = one M2KR passage. **Page = `passage_id` minus its section suffix** (`WikiWeb_<title>_<n>`). Primary recall is at page level. KB capped at ≤ 2,000 pages per dataset. | Checked on the M2KR passages today [V-run]. Without this, doc-ID recall is ambiguous. |
| 7 | **Dev splits** | "A small dev split" (MMQA only defined) | E-VQA: 50 questions from M2KR `EVQA_data/valid`, with gold pages disjoint from the test KB. MMQA: 50 from train. λ, k and the fact threshold are tuned per dataset on dev. | The `valid` split exists [V-run]. The fact threshold had three different defaults (0.6 CLI / 0.2 function / 0.4 smoke guide). |
| 8 | **MMQA role** | Primary home for H1–H3, "images show several entities" | **Secondary, gated (G4).** It has no query image, so the visual query is the question text through SigLIP2. Table questions are skipped. Image gold maps to the same title's passages. | No query image; profile images only; PPR ranks passages only. CLIP text→image is weak on KB-VQA (MG² Table 2). |
| 9 | **H4** | Text-only questions (HotpotQA), modified vs unmodified graph | **MMQA text-only questions over the multimodal KB**, full system vs unmodified MemGraphRAG, non-inferiority margin −3 pp R@5. HotpotQA becomes a bit-identity check. | On HotpotQA there are no images, so H4 could not fail. |
| 10 | **H1 wording** | "beats best *non-graph* visual retrieval", but lists MG²-RAG (a graph method) | "Beats dense CLIP retrieval and MG²-RAG". The baselines that apply are listed per dataset (§5). | MMQA has no image→image baseline. |
| 11 | **Statistics** | "Bootstrap CIs on every comparison" | One primary metric and one primary contrast per hypothesis, using a paired bootstrap and McNemar, with Holm correction. Minimum detectable effect ≈ 5 pp is stated up front. | Prevents fishing. A λ effect of ~1 pt (MG² Table 8) is below what n ≈ 500 can detect. |
| 12 | **Leakage controls** | "with and without query captions" | Additionally: perceptual-hash dedupe of query vs KB images; drop Bing and placeholder KB images; no Wikipedia image captions; H2 judged on retrieval; QA run with equal evidence modality for all systems. | GLDv2 and MuKA images are both mostly Wikimedia, so near-duplicates are possible. MuKA contains 20 Bing thumbnails on E-VQA [V-run]. |
| 13 | **Models** | "Same MLLM for every system" | A role → model table that holds for every system (§4.6). VRAM test of Qwen3-VL-8B on the MIG slice by 7 Oct. | Indexing, answering and judging use different models; the 8B fit on ½ A100 is unmeasured. |
| 14 | **Cost** | 4 indexes, $40–140 with gpt-4o-mini | **All models run on Snellius.** The budget is GPU hours, not dollars (§6.4). Caption variants only index the caption passages (the rest are cache hits). KBs are capped. | The run table needs 7–9 indexes, not 4. The smoke test ran on self-hosted Qwen2.5-7B. |
| 15 | **Claim wording** | "which entities to ground" | "which entities to **localise** (text-prompted concept segmentation)"; native-vs-caption is a secondary analysis; HVM-GraphRAG named in the contrast. | Novelty re-check: HVM-GraphRAG already uses entity types to limit VLM extraction from images. |
| 16 | **Citations** | — | MG²-RAG with its ECCV LNCS DOI `10.1007/978-3-032-37167-6_32`. Add 2511.16654, KBMR (2608.21450), "Signal or Noise?" (2609.35304), PILAR (2609.32895). | Novelty re-check §2–3. |
| 17 | **Fusion vs MG²-RAG** | Audit A4: "MG²-RAG also weights branches, then normalises" | MG²-RAG sums the weighted branches and normalises the **total**. We normalise **each channel**. State this as a difference. | `MMGraphRAG.py:994-1007` [V-read]. |
| 18 | **Datasets ruled out** | CrossModalQA and RETINA as "candidates" | Not usable: CrossModalQA is unreleased; RETINA's HF dataset returns 401. WebQA is not used (51 GB of images, test only via EvalAI). | Data-licence check [V-run]. |
| 19 | **Timeline** | Two competing tables (method.md, audit §6); 2 Oct gate unrecorded | One table with dated gates G1–G6, a pre-committed minimal paper and an extended drop order (§6). | No code and no owners yet. The H3 go/no-go needed data that arrived only after it was due. |
| 20 | **Indexing LLM** | gpt-4o-mini | **Qwen2.5-7B**, self-hosted on Snellius, the same model for every MemGraphRAG variant. G2's histogram must use its output. | This is what ran in the smoke test; it removes API costs. It extracts less than gpt-4o-mini, which is one more reason never to compare with the paper's numbers. |

### 2b. Adopted from the 3 Oct audit (vs the 29 Sep plan)

These were written into method.md on 3 Oct but not yet ratified. They are part of the final method.

| Area | 29 Sep plan | Final |
|---|---|---|
| KB images | "Image nodes linked to passages" on M2KR | M2KR passages are text-only; KB images come from MuKA's Wikimedia URL lists (audit A1) |
| Query images | OVEN/AToMiC, "out of scope" | Ungated `BByrneLab/M2KR_Images` (A19) |
| Dispatch | Visual seeds added inside MemGraphRAG retrieval | Visual seeds computed **before** the fact gate; no dense fallback when visual seeds exist (A3) |
| Seed fusion | `normalise(s_text + λ·s_visual)` | Per-channel L1 normalisation, then `(1−λ)·ŝ_text + λ·ŝ_visual` (A4) |
| H3 | Schema types vs NER labels | Four arms on the same LLM entities (A5) |
| SAM3 prompts | Entity names | Generic noun phrases per visual type (A6) |
| Caption baseline | Generic captions of KB images | Generic **and** entity-aware captions of query **and** KB images, indexed as separate passages (A8) |
| H1 comparison | vs text-only MemGraphRAG | vs dense CLIP and MG²-RAG; text-only is a sanity check (A9) |
| Second dataset | none | MMQA dev (A11; role reduced in 2a #8) |
| Chunking | `index.py` 256-token windows | One passage per chunk, bypassing `index.py` (A13) |
| Crops | Optional region nodes | Not nodes; a crop match seeds its entity (A16) |
| λ = 0 | "text-only" | Renamed "visual graph, no visual seeds" (A10) |
| Naming | "MemGraphRAG" | "MemGraphRAG (released code; Eqs. 7–8 not implemented)"; legacy path "w/o memory (≈ HippoRAG 2 pipeline)" (A17, D5) |
| Encoder facts | 77-token CLIP limit | SigLIP2 truncates at 64 tokens (A12) |
| SAM3 | Vendored copy | transformers `Sam3Model`; acknowledge SAM in the paper (A20) |

---

## 3. Data

### 3.1 E-VQA (primary dataset)

- **Questions:** 500 from M2KR `EVQA_data/test`, stratified by image source (iNat and GLDv2, in proportion to the test set's 2,000 : 1,750). Dev: 50 from `EVQA_data/valid`.
- **KB:**
  - the gold pages of the 500 questions, plus random distractor pages from the 19,267-page M2KR test corpus, ≤ 2,000 pages in total (≈ 2.7 passages per page) [I];
  - chunk = passage; page = `passage_id` without its trailing `_<n>`.
- **KB images:**
  - one per page, from MuKA's `evqa_passages_image_urls.jsonl.gz`;
  - drop Bing thumbnails and black placeholders, and record how many were dropped;
  - download from Wikimedia with a descriptive User-Agent carrying contact details, at most 2 parallel connections, backing off on 429 / Retry-After [V-read].
- **Query images:** `M2KR_Images`. Read iNat members from `inat.zip` with range requests; download the GLDv2 tar (2.8 GB) once.
- **Dedupe:**
  - perceptual-hash every query image against every KB image;
  - exclude questions whose query image near-duplicates a KB image from the primary set;
  - report the rate.
- **Never used:** Wikipedia image captions (`image_reference_descriptions`).
- **Paper wording:** "single-hop, recognition-dominated; one image per page". M2KR has no two-hop items.

### 3.2 MMQA dev (secondary dataset; role decided by gate G4)

- **Image questions:**
  - Use the non-table image questions: ImageQ (230), ImageListQ (141), Compose(TextQ, ImageListQ) (46) and Compose(ImageQ, TextQ) (20) [V-run].
  - That is ≈ 440 of the 940 image questions. Skip types containing TableQ, and say so.
- **Text-only questions for H4:** 300 TextQ questions.
- **KB:** the gold and context documents of the selected questions, ≤ 2,000 documents. One passage per text paragraph; one image node per entity image.
- **Gold:** text gold = document ID. Image gold = the passage(s) with the same Wikipedia title, because PPR ranks passages only.
- **Visual query:** the question text, encoded with SigLIP2 (64 tokens).
- **Gate G4 (by 7 Oct):**
  - On 100 dev image questions, measure SigLIP2 text→image R@5 over the MMQA KB images; also try `Qwen3-VL-Embedding-2B` (32k context).
  - If neither beats the bge text-only passage retrieval's R@5 on the same questions, MMQA is used **only** for H3 grounding precision and H4. Otherwise it also serves H1 and H2.
- **Licence:**
  - None stated by AllenAI. Third-party "Apache-2.0" mirrors carry no authority.
  - Use it for internal research only and do not redistribute.
  - In the paper, write "licence not specified by the authors".
  - Do not show posters or logos in figures (many are non-free Wikipedia files) [V-run, I].

### 3.3 Text regression

200 HotpotQA questions, indexed one passage per chunk from `dataset/hotpotqa/*_corpus.json`. Purpose: a **bit-identity check**. With no visual seeds, our code must reproduce released MemGraphRAG retrievals exactly. It is not a hypothesis test.

### 3.4 Not used

| Dataset | Why not |
|---|---|
| InfoSeek | First to drop: no extra structure, 9 GB tar. Add only if time remains. |
| E-VQA two-hop (official KB) | Fallback only if MMQA fails G4 *and* the group wants a multi-hop result. Needs the 4.9 GB official KB (+≈ 1 day). |
| CrossModalQA, RETINA | Not released (RETINA's dataset returns 401). |
| WebQA | ≈ 51 GB of images; test labels only via EvalAI. |

---

## 4. Method

### 4.1 Indexing (per corpus)

1. Our loader passes one passage per chunk to `index_with_memory`. Never use `code/index.py`.
2. Run MemGraphRAG as released: OpenIE → schema → ontology filter (default `percentile`) → conflict detection and resolution → memory graph. Qwen2.5-7B (self-hosted, via the team's patch), temperature 0.
3. Add one **image node** per KB image, with an edge to every passage of its page. This step is identical for every arm.
4. Run grounding for the H3 arm being evaluated (§4.4).
5. Embed images and crops with SigLIP2-so400m, stored apart from the bge text vectors.

### 4.2 Retrieval (per question)

1. **Text seeds:** exactly as released (facts and passages via bge).
2. **Visual seeds:**
   - The visual query is the query image on E-VQA and the question text on MMQA.
   - The top-k image nodes seed themselves, with weight equal to the cosine score.
   - The top-k crops seed their linked entities, with weight split 1/k across linked entities.
   - k is set on dev.
3. **Fusion:**

   ```
   s = (1 − λ) · ŝ_text + λ · ŝ_visual,   ŝ = s / ‖s‖₁,   ŝ = 0 if ‖s‖₁ = 0
   ```

   λ is tuned on dev only.
4. **Dispatch:**
   - **Visual seeds empty:** run MemGraphRAG exactly as released, including its dense fallback (`MemGraphRAG.py:1100-1104`). Text behaviour is then bit-identical.
   - **Visual seeds present:** compute them first. If no fact passes the threshold, run PPR on passage + visual seeds instead of falling back to dense retrieval.
   - Relax the `sum(node_weights) > 0` assertion (`:2175`).
5. **Fact threshold:** one value per dataset, chosen on dev. Log the share of questions on each dispatch branch.
6. **Generation:** pass the top-5 passages to the answerer (§4.5).

### 4.3 Hypotheses and primary contrasts

| H | Claim | Primary contrast (metric) | Data | Secondary |
|---|---|---|---|---|
| H1 | Visual seeds through the graph beat dense CLIP retrieval and MG²-RAG | Ours (full) vs dense SigLIP2 image→image (page R@5) | E-VQA (+ MMQA if G4 passes, vs SigLIP2 text→image) | vs MG²-RAG (same encoder); vs text-only MemGraphRAG (sanity) |
| H2 | Native visual evidence beats captions | Ours vs entity-aware caption MemGraphRAG (page R@5) | E-VQA (+ MMQA if G4 passes) | vs generic captions; each with and without query-image captions; QA accuracy |
| H3 | Type-guided selection localises the right entities | Type-guided vs NER-label arm (**grounding precision**) | E-VQA and MMQA, per dataset | Coverage; page R@5 across all four arms |
| H4 | The visual layer does not hurt text questions | Full system vs unmodified MemGraphRAG on MMQA TextQ (R@5, non-inferiority, margin −3 pp) | MMQA TextQ | HotpotQA bit-identity check |

### 4.4 Grounding arms (H3)

**Candidates for image *i*:** entities that occur in the passages of *i*'s page and have a node in the final graph.

**Visual types and noun phrases:**
- Decide once per distinct type whether it is *visual*: one LLM yes/no call per type, reviewed by one person.
- Map each type to a generic noun phrase ("plant", "bird", "building", "painting"…).
- Types without a mapping use their lowercased label.
- If an entity has several types, use its visual type if it has one, otherwise its most frequent type.
- Publish the type list and the mapping in the repo.

**Arms** (all prompt SAM3 with the candidate's noun phrase):

| Arm | Selected candidates |
|---|---|
| none | — |
| all | every candidate |
| NER-label (MG²-style) | candidates whose matching spaCy `en_core_web_trf` span in the same passage is labelled FAC, LOC, ORG, PERSON, PRODUCT or WORK_OF_ART. No matching span means the candidate is not selected. |
| type-guided | candidates with a visual type. If **D7 fires**, the types come from the fine-grained `entity_type_extraction` prompt (+1 LLM call per chunk), and this arm cannot then be dropped. If D7 does not fire, the fine-grained version is an optional fifth arm. |

**SAM3 settings:** keep masks above SAM3's default score threshold, at most 5 per noun phrase per image.

**Crop → entity link:** if the page's title entity is among the candidates with that noun phrase, link the crop to it. Otherwise link it to all of them, with weight 1/k.

**Measurement:**
- **Precision:** 100 random crops per arm per dataset. Two annotators, blind to arm, judge whether "the crop shows the entity it is linked to". Report Cohen's κ and a Wilson 95 % CI.
- **Coverage:** the share of KB images whose page-title entity gets at least one linked crop.
- **Filter loss:** the number of visual-typed entities the ontology filter removed.

### 4.5 Baselines and ablations

| Run | Role | E-VQA | MMQA |
|---|---|---|---|
| MemGraphRAG-V (full, type-guided) | ours | ✓ | ✓ |
| Caption MemGraphRAG: generic captions of query + KB images | H2 baseline | ✓ | KB images only (no query image) |
| Caption MemGraphRAG: entity-aware captions | H2 baseline (strong) | ✓ | KB images only |
| Text-only MemGraphRAG (released code) | sanity / H4 | ✓ | ✓ |
| Dense SigLIP2: image→image, image→text | H1 baseline | ✓ | text→image only |
| Zero-shot MLLM (no retrieval) | floor | ✓ | ✓ |
| MG²-RAG with SigLIP2 | H1 baseline | ✓ | optional (dropped early) |
| H3 arms: none / all / NER-label / type-guided | H3 | ✓ | ✓ |
| Visual graph, no visual seeds (λ = 0) | ablation | ✓ | ✓ |
| w/o memory (legacy `index()`, ≈ HippoRAG 2 pipeline) | ablation | ✓ | — |

**QA evidence modality:** run QA twice:
- every system gets text passages only;
- every system also gets the images of its retrieved pages.

H2's primary result is the retrieval metric, so a gain cannot come from feeding the generator more input.

### 4.6 Models (the same for every system)

| Role | Model | Note |
|---|---|---|
| Indexing LLM (all MemGraphRAG variants) | Qwen2.5-7B, self-hosted, temperature 0 | Runs as a separate job from Qwen3-VL (the two don't fit on one MIG slice together). Use one version (precision, quantisation) for every system. |
| Text embedder | `bge-large-en-v1.5`, released wrapper (mean pooling, instruction dropped) | D6: keep as released and state it |
| Visual encoder | SigLIP2-so400m, also inside MG²-RAG | D3; EVA-CLIP-8B not used |
| Segmenter | SAM3 via transformers, weights downloaded once with the approved HF token and run on Snellius | Access granted (G3). Do not use HF hosted inference: it is paid and sends images off-site. OWLv2 only as a backup |
| Captioner and answerer | Qwen3-VL-8B | D4: Qwen3-VL-4B if the 7 Oct VRAM test on the MIG slice fails |
| Judge | Containment (primary, no model) + LLM judge: Qwen2.5-7B on Snellius, or gpt-4o-mini (≈ $1–3 for ≈ 25k calls) | The judge must differ from the answerer (Qwen3-VL), because a model grading its own answers tends to be lenient. BEM if TensorFlow installs on Snellius |
| NER (NER arm; MG²-RAG text side) | spaCy `en_core_web_trf` | |

MemGraphRAG's Eqs. 7–8 stay unimplemented (D5).

### 4.7 Metrics and statistics

- **Retrieval:**
  - page R@{1, 5, 10} and MRR (primary: page R@5);
  - passage R@5 as a secondary measure.
- **QA:**
  - containment accuracy and LLM-judge accuracy;
  - for E-VQA, BEM if available;
  - call InfoSeek "InfoSeek val (M2KR test split)" if it is used.
- **Tests:**
  - paired bootstrap (10,000 resamples) for 95 % CIs on R@5 differences;
  - McNemar for accuracy differences;
  - Holm correction across the four primary contrasts.
- **Power:** with n ≈ 500, the minimum detectable effect is ≈ 5 pp [I]. Report smaller differences as "not detectable", not as "no effect". λ and k sweeps are descriptive only.
- **Comparability:** never put our numbers in a table with either paper's published numbers. Every baseline is re-run on our subsets.

---

## 5. Which baseline answers which hypothesis on which dataset

| | E-VQA | MMQA (G4 passes) | MMQA (G4 fails) |
|---|---|---|---|
| H1 | dense image→image, MG²-RAG | dense text→image | — |
| H2 | both caption baselines | both caption baselines (KB side) | — |
| H3 | precision, coverage, R@5 | precision, coverage, R@5 | precision, coverage |
| H4 | — | TextQ non-inferiority | TextQ non-inferiority |

---

## 6. Plan, gates and budget

### 6.1 Gates

| Gate | Date | Test | If it fails |
|---|---|---|---|
| G1 | ✅ passed | A real MemGraphRAG index + QA run on Snellius (task 01): done with Qwen2.5-7B | — Record wall-clock time, chunk count and tokens in task 01's log; §6.4 depends on them |
| G2 | 7 Oct | **Schema-type histogram on 50 E-VQA passages** (hand-built from the parquet; schema extraction only; no images needed) | Mostly OntoNotes labels → D7 fires: the type-guided arm uses fine-grained types |
| G3 | ✅ passed | SAM3 access approved | — (OWLv2 stays as a backup only if SAM3 fails technically) |
| G4 | 7 Oct | MMQA SigLIP2 / Qwen3-VL-Embedding text→image check (§3.2) | MMQA only for H3 precision and H4 |
| G5 | 7 Oct | Qwen3-VL-8B + KV cache + images fit on the MIG slice | Qwen3-VL-4B |
| G6 | **11 Oct** | Visual layer runs end-to-end on E-VQA dev | Switch to the minimal paper (§6.3) |

### 6.2 Timeline (replaces method.md and audit §6)

| Dates | Work |
|---|---|
| 4 Oct | Call: ratify, assign owners, record the smoke-test status |
| 4–7 Oct | G2, G4, G5; type → noun-phrase list draft; SAM3 prototype on a few E-VQA images; measure indexing throughput (§6.4) |
| 5–8 Oct | Data builders: E-VQA (MuKA images, dedupe, page grouping) and MMQA subset; per-passage loader; metrics and statistics scripts |
| 5–11 Oct | Visual layer: dispatch, per-channel fusion, image nodes, grounding arms, crop seeds |
| 8–12 Oct | Caption indexing starts **8 Oct**; dense SigLIP2 and zero-shot (8–10 Oct); MG²-RAG with SigLIP2 (9–12 Oct) |
| 9–12 Oct | Grounding-precision annotation (2 annotators) |
| 12–16 Oct | Main runs and ablations; λ, k and threshold on dev. **Results freeze 16 Oct.** |
| 5–22 Oct | Writing (background and method from 5 Oct) |
| 23 Oct | Submission (9-page ACM) |

### 6.3 Cuts

- **Minimal paper** (if G6 fails): E-VQA only, with
  - H1 vs dense SigLIP2,
  - H2 vs both caption baselines,
  - H3 as grounding precision and coverage only.
- **Drop order:** λ sweep → InfoSeek → MG²-RAG on MMQA → w/o-memory ablation → fine-grained arm (only if D7 did not fire) → MG²-RAG on E-VQA.
- **Never drop:** caption baselines, H3 grounding precision, H4.

### 6.4 Compute budget [I]

Everything runs on Snellius. The budget is **GPU hours**, not dollars.

**Partitions (course guide):**
- `int3`: free, 1/7 A100, for testing only;
- `gpu_mig` with the course reservation: free (TU/e pays), ½ A100;
- `gpu_mig` without the reservation: 64 credits/h;
- `gpu_a100`: 128 credits/h.

Check the group's balance with `accinfo`.

**Indexes to build (Qwen2.5-7B):**

| Index | Count | Note |
|---|---|---|
| Base multimodal KB (shared by ours, H3 arms, text-only, λ = 0) | 2 (E-VQA, MMQA) | ≈ HotpotQA size each (≈ 5,400 E-VQA passages); grounding adds no LLM calls |
| + generic captions, + entity-aware captions | 4 | Only the caption passages miss the SQLite cache |
| w/o memory (`index()`) | 1 (E-VQA) | |
| Fine-grained types | +1 call per chunk on the 2 base KBs | Only if D7 fires |
| HotpotQA regression | 1 | 200 questions |

**Size of the work:** a HotpotQA-size index was estimated at 66k–142k LLM calls and 60–180M input tokens (`audit/evidence/indexing_cost_estimate.json`). On ½ A100, the 7B weights (~15 of ~20 GB) leave little room for batching. A rough guess, not measured, is 5–15 GPU-hours per base index, and a few dozen GPU-hours in total. This is feasible on the free reservation, but it is the critical path.

**To keep it safe:**
1. **Measure.** Take the smoke test's wall-clock time and chunk count from task 01's log, and replace this estimate by 7 Oct.
2. **Use a quantised Qwen2.5-7B** (AWQ or FP8) to free memory for batching. Use the same version for every system.
3. **Turn on vLLM prefix caching.** Every call shares a 200–700-token fixed instruction.
4. **Run Qwen2.5-7B (indexing) and Qwen3-VL-8B (captions, answers) as separate jobs and phases.** They don't fit on one slice together.
5. **If indexing falls behind, run only the base indexes on `gpu_a100`.** For example, 10 h = 1,280 credits.
6. **Download data, images and weights once from the login node** into scratch or project space. Check whether compute nodes have internet access before a job depends on it.
7. **Respect Wikimedia's download limits.** Use a descriptive User-Agent, at most 2 parallel connections, and back off on 429. MuKA images take time, not money.

**External costs:** none required. If the judge is gpt-4o-mini, ≈ 25k short calls ≈ $1–3.

---

## 7. For the 4 Oct call

**Ratify (defaults adopted in this document):**

- [ ] D1: E-VQA primary + MMQA secondary, gated by G4; InfoSeek optional.
- [ ] D2: claim as worded in §1.
- [ ] D3: SigLIP2 everywhere, including MG²-RAG.
- [ ] D4: Qwen3-VL-8B (4B fallback) as captioner and answerer for every system.
- [ ] D5: no Eqs. 7–8.
- [ ] D6: BGE wrapper as released.
- [ ] D7: G2 histogram rule.
- [x] D8: resolved; SAM3 access granted (OWLv2 is a backup only).
- [ ] New: H3's primary outcome is grounding precision (2a #1).
- [ ] New: H4 on MMQA TextQ with a −3 pp margin (2a #9).
- [ ] New: grounding adds no edges (2a #4).

**Decide:**

- [ ] Compute plan: free MIG reservation by default. How many `gpu_a100` credits may we spend on indexing if it falls behind? Check `accinfo` first.
- [ ] Judge: local Qwen2.5-7B ($0) or gpt-4o-mini (≈ $1–3).
- [ ] The G4 threshold for MMQA, as proposed in §3.2.
- [ ] Use of MMQA without a stated licence (internal research use, no redistribution).
- [ ] An owner per workstream: smoke test / G2; data builders; visual layer; baselines (captions, dense, zero-shot, MG²-RAG); metrics and statistics; annotation; writing.

**Find out:**

- [x] Smoke test: passed with Qwen2.5-7B. Still to do: record the patch, the sampling override, wall-clock time, chunk count and the schema-type output in [task 01's log](tasks/01-snellius-smoke-test.md#log).
- [x] SAM3 access: granted.

After the call, merge PR #1 together with this document. Then update the README status line, and mark method.md as background, superseded by this document where they differ.
