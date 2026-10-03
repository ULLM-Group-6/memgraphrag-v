# Research findings: methodology and plan vs primary sources (3 Oct 2026)

Scope: checks `docs/method.md`, `docs/audit/audit-report.md` and `docs/tasks/03-04` against the papers, model/dataset cards, official docs and code. Tags as in the audit: **[V-read]** verified by reading the primary source at the cited location, **[V-run]** verified by running something (data statistics computed on downloaded files, HTTP range reads), **[I]** inferred. Every paper listed was opened at the given URL (arXiv/ACL Anthology/CVF/ECCV) or verified through the arXiv or Semantic Scholar API. Downloads used for checks (all small, in a scratch directory, not in the repo): M2KR `EVQA_data/test`, `Infoseek_data/test`, `EVQA_passages/test` parquet files (≈55 MB), MuKA URL lists (2 MB), MMQA dev/images metadata (4 MB), paper PDFs.

---

## 1. Summary

| # | Finding | Severity | Recommended change |
|---|---|---|---|
| 1 | **The M2KR knowledge-base side is text-only.** `EVQA_passages` has only `language, passage_id, passage_content`; `Infoseek_passages` has `passage_id, passage_content, title` [V-read dataset card; V-run parquet]. On M2KR InfoSeek/E-VQA the only image is the query image, so "image nodes linked to passages" and "grounding entities in KB images" have nothing to index unless KB images are added from another source. | **Blocker** | Add KB images from **MuKA's released image-URL lists** (one Wikimedia image per entity; covers 51,472/51,472 M2KR E-VQA test passages and ≈4,638/4,708 InfoSeek gold entities [V-run]) or from the **official E-VQA KB** (`encyclopedic_kb_wiki.zip`, 4.9 GB, several `image_urls` per article with `image_section_indices`, so images can be attached to sections) [V-read]. Write this into `method.md` §Data. |
| 2 | **M2KR's E-VQA split has no two-hop questions.** Its 3,750 test items are 1,000 `templated` + 2,750 `automatic`, one gold passage each [V-run]. The official E-VQA test set has 5,750 items including 1,000 two-hop and 1,000 multi-answer (E-VQA paper Table 2) [V-read]. The plan's fallback "prefer E-VQA two-hop items" is impossible from M2KR. | **Blocker** (for the memory/multi-hop story) | If multi-hop matters, take two-hop questions from the official `test.csv` plus the official KB (≈1 extra day). Otherwise state plainly that the KB-VQA sets are single-hop. |
| 3 | **H3 is at risk: MemGraphRAG's "schema types" use the same label set as MG²-RAG's spaCy NER.** The schema prompt (`code/src/prompts/prompt.py:308-326`, used at `MemGraphRAG.py:381`) lists exactly the 18 OntoNotes labels (`<CARDINAL>`…`<WORK_OF_ART>`) as "common types". MG²-RAG filters spaCy `en_core_web_trf` entities to 6 of those same labels (`MMGraphRAG.py:582`). "Animal" is not in the list. A fine-grained type prompt with ~200 types (Species, Building, Bridge, Vehicle, Artwork…) exists in the same file (`prompt.py:2-295`) but is **never called** [V-read]. | **High** | In week 1, count the schema types MemGraphRAG actually produces. Make H3 a three-way comparison: spaCy 6-label filter vs MemGraphRAG schema types vs fine-grained types from the unused `entity_type_extraction` prompt. Reword the claim to say what really differs: *which* entities are prompted (LLM triple entities, including common nouns) and *how* types become SAM3 prompts. |
| 4 | **On E-VQA/InfoSeek the query image does all the work; question text is nearly useless for retrieval.** 3,710/3,750 E-VQA and 4,443/4,708 InfoSeek M2KR test questions are deictic ("this plant…"). The gold entity's name appears in 34 and 0 of them [V-run]. MG² Table 2: EVA-CLIP-8B text→image R@1 is 1.8 (E-VQA) and 0.2 (InfoSeek), while image→image is 30.0/47.2 and image→text 42.0/56.5 [V-read]. With one entity photo per page, SAM3 crops add little [I]. | **High** | (a) H1 against text-only MemGraphRAG is a foregone conclusion: keep it as a sanity check, not a result. (b) Feed the *query image* (not the question text) to the visual seed channel. (c) Change the retrieval dispatch so visual seeds are added before MemGraphRAG's dense fallback (`MemGraphRAG.py:1100-1105`); otherwise almost every E-VQA question skips the graph. (d) Test H1/H3 on a corpus with multi-entity images and text questions (finding 6). |
| 5 | **The caption baseline must caption the query image as well, ideally entity-aware, and it may well win.** E-VQA paper: Google Lens entity recognition + KB section reaches 48.8% (PaLM) vs 87.0% with oracle sections (Tab. 5) [V-read]. TrioRAG (2609.19417) finds VLM-generated text queries more robust than image retrieval [V-read abstract]. A 2026 controlled study (2607.16604) warns that apparent multimodal gains often come from textual leakage [V-read abstract]. | **High** | Run two caption baselines (generic caption; entity-aware caption that names the species/landmark), both captioning query *and* KB images with the same MLLM used for answering. Decide up front whether E-VQA KB image captions (`image_reference_descriptions`) may be used. Add RAG-Anything (2510.12323) as an off-the-shelf caption-centric multimodal graph RAG baseline if time allows. |
| 6 | **The novelty margin is narrow.** MG²-RAG (now **ECCV 2026**) already does graph RAG + image nodes + SAM3 crops linked to entities + PPR with visual seeds. HVM-GraphRAG (2607.24861) already combines conflict-detecting/resolving graph construction with images (via VLM summaries). EviProp (2606.08979) seeds PPR with visual priors. No paper found that combines memory/conflict-resolved graph RAG with **type-guided** visual grounding [V-read; search details in §2]. | **High** | Narrow the claim to: *does LLM-derived entity typing from a conflict-resolved memory graph improve visual grounding coverage and retrieval over NER-label filtering, and does native visual evidence beat captions under leakage controls?* Present it as a controlled study on top of two existing systems, not a new architecture. Cite MG²-RAG as ECCV 2026. |
| 7 | **Query images are not locked behind 294 GB / 174 GB.** `BByrneLab/M2KR_Images` (ungated) hosts E-VQA query images (`inat.zip` 8.92 GB + `google-landmark.tar` 2.79 GB) and InfoSeek val images (`infoseek_val_images.tar` 8.96 GB). I pulled a single image from `inat.zip` with HTTP range requests (≈19.5 MB read in total) [V-run]. The OVEN and AToMiC HF repos are `gated=auto`, not manual [V-run, audit evidence file]. | Medium (good news) | Update audit §1.4/§7 and the `method.md` Data paragraph. Budget ≈3 GB plus streaming one 9 GB tar on Snellius for InfoSeek. |
| 8 | **SAM3 is built for short generic noun phrases**, not proper names: its text encoder has a 32-token context, "struggles to generalize to fine-grained out-of-domain concepts" (App. B), and is "constrained to simple noun phrase prompts" [V-read 2511.16719]. MG²-RAG prompts SAM3 with normalised entity names such as "hinrich lichtenstein" [V-read]. | Medium | Map types to generic noun phrases (`<Species>`→"animal"/"plant", Building→"building") and link masks to the page's entity, instead of prompting entity names. SAM3 is in HF transformers (`Sam3Model`), weights gated with manual approval. The SAM License allows research use and redistribution (with the licence attached) and **requires acknowledging SAM in publications** (§1.b.ii) [V-read]. |
| 9 | **Encoders.** SigLIP2's text side truncates at **64** tokens, not 77 (transformers SigLIP2 docs; `SiglipTextConfig` default 64) [V-read]. EVA-CLIP-8B: 8.1B params, 77-token text, HF weights = 4 fp32 shards totalling 32.9 GB, `trust_remote_code` [V-read]. MG² has **no encoder ablation** [V-read]. | Medium | Fix `method.md` design choice 1. Swapping the encoder changes results relative to MG²-RAG, so either run MG²-RAG with the same encoder or report EVA-CLIP-8B for the MG² comparison. Consider `Qwen/Qwen3-VL-Embedding-2B` (Apache-2.0, 4.3 GB, text+image, 32k context) for joint image+question queries. |
| 10 | **Our numbers will not be comparable to either paper.** MG² uses a 100k-document KB (Table 2) or a 5k KB with guaranteed evidence (Tables 3, 6), the full E-VQA test set and 5.8k InfoSeek-val samples, with BEM / VQA accuracy. MemGraphRAG uses NV-Embed-v2, gpt-4o-mini and 1,000 validation questions per dataset [V-read]. | Medium | Never put our subset numbers in the same table as published ones; re-run every baseline on our subset. Call the text system "MemGraphRAG (released code)": its retrieval is HippoRAG 2's seed rule, PPR damping 0.5 and passage weight 0.05 (HippoRAG 2 §3.5, §6.2, App. table) [V-read]. |

Lower-severity items: gpt-4o-mini is still listed at $0.15 / $0.60 per 1M input/output tokens ($0.075 / $0.30 Batch) and is **not** on OpenAI's deprecations page, so the $13–37 estimate holds; avoid `gpt-4.1-nano` (shutdown 23 Oct 2026) (§4.6). Qwen3-VL-8B and Qwen3.5-9B (Apache-2.0) are newer captioner options than Qwen2.5-VL-7B (§4.5). Neither base paper has a new arXiv version and neither repo has commits after 28 Sep 2026 (§2.3).

---

## 2. Novelty assessment and related work

### 2.1 Forward citations (Q1)

Semantic Scholar Graph API, queried 3 Oct 2026 [V-run]. Its coverage of very recent papers lags; Google Scholar was not checked.

| Base paper | Citing paper | Overlap with our idea? |
|---|---|---|
| MemGraphRAG (arXiv 2606.00610; 1 citation) | *Towards Trustworthy and Cost-Efficient Data Integration: From Naïve RAG to Agentic RAG*, IEEE Data Eng. Bull. (to appear), [arXiv 2607.22319](https://arxiv.org/abs/2607.22319) | No. A vision/survey paper on RAG for data integration. |
| MG²-RAG (arXiv 2604.04969; 2 citations) | *Very Efficient Listwise Multimodal Reranking for Long Documents* (ZipRerank), ICML 2026, [arXiv 2605.11864](https://arxiv.org/abs/2605.11864) | No. A reranker for document pages. |
| | *Position: Text Embeddings Should Capture Implicit Semantics…*, ICML 2026, [arXiv 2506.08354](https://arxiv.org/abs/2506.08354) (v2, May 2026) | No. |

### 2.2 Related work table (Q2, Q3)

Verdicts: **cite** (background), **compare** (should be a baseline or discussed in results), **threat** (overlaps the claimed novelty). Venues come from the arXiv comment field, the paper itself, or the Semantic Scholar venue field (S2), which gives the venue but not always the year.

| Paper | Venue, year | Link | What it does (1 line) | Verdict | Why |
|---|---|---|---|---|---|
| MG²-RAG | ECCV 2026 | [2604.04969](https://arxiv.org/abs/2604.04969); [poster](https://eccv.ecva.net/virtual/2026/poster/3411) | spaCy text graph + SAM3 entity-driven object crops + EVA-CLIP nodes + PPR with text and visual seeds | **threat** (base system, closest prior art) | Already has image nodes, crops linked to entities, visual PPR seeds. Our delta must be stated against it: memory/conflict-resolved LLM graph and type-guided prompt selection. |
| MemGraphRAG | KDD 2026 | [2606.00610](https://arxiv.org/abs/2606.00610) | Schema/fact/passage memory; LLM conflict detection and resolution; PPR retrieval | base | Text backbone. |
| HVM-GraphRAG | arXiv, Jul 2026 | [2607.24861](https://arxiv.org/abs/2607.24861) | "Holistic view" of accepted facts with LLM conflict detection and resolution while building a multimodal document graph; images turned into VLM summaries; concept-level retrieval; MMLongBench, M3DocVQA, Qasper | **threat (partial)** + compare | Already "conflict-aware memory graph RAG + images", but via captions on document QA, without grounding or visual seeds. Undercuts any broad claim of "first memory graph RAG with images". Supports our caption baseline as the natural competitor. |
| mKG-RAG | SIGIR 2026 | [2508.05318](https://arxiv.org/abs/2508.05318) | MLLM-built multimodal KG: scene-graph objects (EGTR) matched to text entities by an MLLM; dual-stage retrieval; E-VQA and InfoSeek, 100k KB | compare | Direct KB-VQA competitor (MG² compares against it). Region–entity alignment, but no type-guided prompting and no PPR. |
| MMGraphRAG | arXiv (v3 Jul 2026; venue not verified) | [2507.20804](https://arxiv.org/abs/2507.20804) | YOLO segmentation + MLLM region descriptions + SpecLink cross-modal entity linking; DocBench, MMLongBench | cite | Region-level image graph, but class-agnostic segmentation and MLLM linking. Weak on E-VQA 5k in MG² Table 3 (16.52 BEM, All). |
| VaLiK | ICCV 2025 | [2503.12972](https://arxiv.org/abs/2503.12972) | Cascaded VLMs turn images into descriptions; cross-modal similarity check; MMKG from descriptions | cite | A "translation-to-text" MMKG. In MG² Table 3 it scores 15.22 (E-VQA 5k, All) vs MG² 53.36 with the same Qwen2.5-VL-7B. |
| RAG-Anything | arXiv, 2025 | [2510.12323](https://arxiv.org/abs/2510.12323) | Multimodal documents; VLM descriptions of images become "multimodal entity nodes" in a dual graph; graph traversal + dense matching | compare | Off-the-shelf caption-centric multimodal graph RAG with image nodes. A reviewer may ask for it. |
| MegaRAG | ACL 2026 | [2512.20626](https://arxiv.org/abs/2512.20626) | MLLM-extracted multimodal KG for long visual documents | cite | Document QA, no grounding. |
| M³KG-RAG | CVPR 2026 | [2512.20136](https://arxiv.org/abs/2512.20136) | Multi-hop audio-visual MMKG built by a multi-agent pipeline; GRASP grounding and pruning | cite | Different modality focus (audio-visual). |
| Context-Enhanced MMKG (CEMMKG) | arXiv, Aug 2026 | [2608.25986](https://arxiv.org/abs/2608.25986) | Adds local/global text context to MLLM image-to-graph extraction; MMLongBench-Doc | cite | Shares the idea "text should guide visual extraction", but through MLLM prompts; no detection or types. |
| EviProp | arXiv, Jun 2026 | [2606.08979](https://arxiv.org/abs/2606.08979) | PPR over chunk–page graphs seeded by dense visual page priors + sparse chunk seeds | cite | Prior art for "visual evidence as PPR seeds" (besides MG²). |
| DualG-MRAG | ACM MM 2026 | [2607.28580](https://arxiv.org/abs/2607.28580) | Macro/micro graphs with a GNN retriever for multimodal multi-hop RAG | cite | Related graph MM-RAG. |
| GraphLoom | arXiv, Aug 2026 | [2608.15056](https://arxiv.org/abs/2608.15056) | Instance-level MMKG from scene descriptions; reliability-aware subgraph routing; ScienceQA, MultiModalQA, OK-VQA | cite | Useful MMQA reference point if MMQA is adopted. |
| EvoGraph-R1 | CVPR 2026 | [2607.12764](https://arxiv.org/abs/2607.12764) | Agentic, self-evolving multimodal hypergraph retrieval (RL) | cite | Agentic; different paradigm. |
| LILaC | EMNLP 2025 (per project page) | [2602.04263](https://arxiv.org/abs/2602.04263) | Layered component graph for open-domain multimodal multi-hop retrieval | compare (only if MMQA/WebQA used) | Multi-hop multimodal graph retrieval. |
| TrioRAG ("Less Is More") | arXiv, Sep 2026 | [2609.19417](https://arxiv.org/abs/2609.19417) | Graph-free late fusion of question, image and VLM-generated query; image retrieval reaches 19.3% document recall on out-of-corpus images | cite | Evidence that text derived from the image (by a VLM) is a strong signal, which strengthens the caption baseline. |
| Controlled Evaluation of Graph and Multimodal Augmentation in RAG | arXiv, Jul 2026 | [2607.16604](https://arxiv.org/abs/2607.16604) | Stage-controlled study: KG used during retrieval helps; multimodal gains are sensitive to textual leakage | cite | Methodology warning for H2. |
| HuLiRAG | arXiv, 2025 | [2510.10426](https://arxiv.org/abs/2510.10426) | Query-time open-vocabulary detection + SAM masks to anchor MLLM RAG | cite | Grounding at query time, not KG construction. |
| PixSearch | arXiv, 2026 | [2601.19060](https://arxiv.org/abs/2601.19060) | Segmenting LMM that emits masks as retrieval queries | cite | Region-level retrieval. |
| VimRAG | arXiv, 2026 | [2602.12735](https://arxiv.org/abs/2602.12735) | "Multimodal memory graph" of agent reasoning states (RL) | cite | Name collision only; it is not a KG. |
| GraMRAG | arXiv, Sep 2026 | [2609.14066](https://arxiv.org/abs/2609.14066) | Multi-agent multimodal RAG with a dynamic graph memory of actions/evidence (RL) | not relevant | Agentic memory, not a corpus graph. |
| HippoRAG | NeurIPS 2024 | [2405.14831](https://arxiv.org/abs/2405.14831) | OpenIE KG + PPR retrieval | cite | Ancestor of both systems. |
| HippoRAG 2 | ICML 2025 | [2502.14802](https://arxiv.org/abs/2502.14802) | Adds passage nodes, recognition-memory triple filter, passage weight 0.05, damping 0.5 | compare | MemGraphRAG's released retrieval follows it (§4.8). |
| GraphRAG (From Local to Global) | arXiv, 2024 | [2404.16130](https://arxiv.org/abs/2404.16130) | Community-summary graph RAG | cite | Background. |
| LightRAG | EMNLP (S2 venue field); arXiv 2024 | [2410.05779](https://arxiv.org/abs/2410.05779) | Dual-level graph retrieval | cite | Background. |
| EchoSight | EMNLP 2024 (S2 venue field) | [2407.12735](https://arxiv.org/abs/2407.12735) | Visual-only search then text reranking; E-VQA 2M KB, InfoSeek 100k KB filtered from E-VQA's KB | compare | MG² follows its E-VQA/InfoSeek setup (MG² repo issue #1 reply). |
| OMGM | ACL 2025 | [2505.07879](https://arxiv.org/abs/2505.07879) | Coarse-to-fine multi-granularity multimodal retrieval for KB-VQA | compare | KB-VQA retrieval SOTA-style baseline; MG² uses its InfoSeek prompts. |
| ReflectiVA | CVPR 2025 | [2411.16863](https://arxiv.org/abs/2411.16863) | MLLM with reflective tokens for KB-VQA | cite | |
| Wiki-LLaVA | CVPRW 2024 | [2404.15406](https://arxiv.org/abs/2404.15406) | Hierarchical retrieval from a Wikipedia KB for MLLMs | cite | |
| mR2AG | arXiv, 2024 | [2411.15041](https://arxiv.org/abs/2411.15041) | Retrieval-reflection for KB-VQA (InfoSeek, E-VQA) | cite | |
| PreFLMR / M2KR | ACL 2024 | [2402.08327](https://arxiv.org/abs/2402.08327) | Late-interaction multimodal retriever; M2KR benchmark suite | cite (data source) | |
| FLMR | NeurIPS 2023 | [2309.17133](https://arxiv.org/abs/2309.17133) | Fine-grained late-interaction multimodal retrieval | cite | |
| MuKA | COLING 2025 | [ACL Anthology 2025.coling-main.647](https://aclanthology.org/2025.coling-main.647/) | Pairs one entity image with each M2KR InfoSeek/E-VQA passage to make a multimodal KB | cite + **use its data** | Released image URLs solve finding 1 ([repo](https://github.com/lhdeng-gh/MuKA), `data_preparation/`). |
| RETINA ("Breaking the Visual Shortcuts…") | arXiv, Nov 2025 | [2511.22843](https://arxiv.org/abs/2511.22843) | Shows E-VQA/InfoSeek queries match the target document's main entity ("visual shortcut"); builds a benchmark with related-entity images | cite | Explains finding 4; candidate benchmark, but release not verified. |
| CrossModalQA | arXiv, Aug 2026 | [2609.05518](https://arxiv.org/abs/2609.05518) | 1,863 cross-modal multi-hop QAs over 4,987 Wikipedia articles + 4,431 Commons images; avg 3.5 hops; gold subgraphs | cite / benchmark candidate | Ideal fit, but data release not verified (§6). |
| MKG-RAG-Bench | KDD 2026 | [2606.26458](https://arxiv.org/abs/2606.26458) | Benchmark for retrieval in multimodal-KG RAG ([GitHub](https://github.com/XiaochenWang-PSU/MKGRAG-Bench)) | cite | KG-native benchmark; not checked in detail. |
| E-VQA / InfoSeek / OVEN | ICCV 2023 / EMNLP 2023 / ICCV 2023 | [2306.09224](https://arxiv.org/abs/2306.09224), [2302.11713](https://arxiv.org/abs/2302.11713), [2302.11154](https://arxiv.org/abs/2302.11154) | KB-VQA datasets and entity recognition | cite | |
| WebQA / MultimodalQA / ManyModalQA / Dyn-VQA / MRAG-Bench / ViDoRAG (ViDoSeek) | CVPR 2022 / ICLR 2021 / AAAI 2020 / ICLR 2025 / ICLR 2025 / EMNLP 2025 | [2109.00590](https://arxiv.org/abs/2109.00590), [2104.06039](https://arxiv.org/abs/2104.06039), [2001.08034](https://arxiv.org/abs/2001.08034), [2411.02937](https://arxiv.org/abs/2411.02937), [2410.08182](https://arxiv.org/abs/2410.08182), [2502.18017](https://arxiv.org/abs/2502.18017) | Multimodal (multi-hop) QA / RAG benchmarks | cite | See §3.3. |
| VisRAG / ColPali / M3DocRAG / UniversalRAG / HM-RAG / MuRAG | ICLR 2025 / ICLR 2025 / arXiv 2024 / ACL 2026 / ACM MM 2025 / EMNLP 2022 | [2410.10594](https://arxiv.org/abs/2410.10594), [2407.01449](https://arxiv.org/abs/2407.01449), [2411.04952](https://arxiv.org/abs/2411.04952), [2504.20734](https://arxiv.org/abs/2504.20734), [2504.12330](https://arxiv.org/abs/2504.12330), [2210.02928](https://arxiv.org/abs/2210.02928) | Multimodal / visual-document RAG | cite | Background. VisRAG reports a 20–40% end-to-end gain over text-based (parsing) RAG on documents. |
| SAM 3 / SigLIP 2 / EVA-CLIP-18B / Qwen2.5-VL / Qwen3-VL / Qwen3-VL-Embedding | arXiv 2025–2026 | [2511.16719](https://arxiv.org/abs/2511.16719), [2502.14786](https://arxiv.org/abs/2502.14786), [2402.04252](https://arxiv.org/abs/2402.04252), [2502.13923](https://arxiv.org/abs/2502.13923), [2511.21631](https://arxiv.org/abs/2511.21631), [2601.04720](https://arxiv.org/abs/2601.04720) | Models used | cite | |
| Visual-RFT | ICCV 2025 | [2503.01785](https://arxiv.org/abs/2503.01785) | RL fine-tuning of LVLMs for perception tasks | not relevant | No retrieval or KG. |

The prompt also listed "Graph-based multimodal RAG for KB-VQA". The only graph-based KB-VQA systems found are MG²-RAG and mKG-RAG.

### 2.3 Base-paper versions and repo activity (Q4)

- **MemGraphRAG**: arXiv still **v1** (30 May 2026); the comment "Accepted by KDD 2026" and the DOI 10.1145/3770855.3818074 were already in v1 [V-run arXiv API, V-read paper p.1].
- **MG²-RAG**: arXiv still **v2** (12 Jul 2026). **Accepted at ECCV 2026**: the repo README banner and BibTeX (`booktitle={…19th European Conference on Computer Vision (ECCV)}`), and the ECCV 2026 poster page 3411 lists the same four authors [V-read]. The repo's README and `method.md` cite it only as arXiv.
- **Commits**: `XMUDeepLIT/MemGraphRAG` latest commit 20 Jun 2026 ("revise code"); `Daboolu/MG2-RAG` latest 31 Aug 2026 (E-VQA builder, docs). **No commits after 28 Sep 2026**, so the audit's findings stand [V-run GitHub API].
- **Independent confirmation from issues** [V-read]: MemGraphRAG issue #6 (28 Jul 2026, unanswered) asks where the paper's IDF passage-initialisation term is in the code, matching audit §2.1. Issue #7 asks about G-Novel results; the dataset is absent from the repo although Table 1 reports it. MG²-RAG issue #1 lists missing 100k KB IDs, loaders, runners and evaluation scripts; the author replied that E-VQA/InfoSeek follow EchoSight and ScienceQA/CrisisMMD follow VaLiK.

### 2.4 Threat analysis and pivot

What the project claims as new (README; `method.md`): (1) memory/graph RAG + native image evidence, (2) ontology-guided grounding (schema types choose SAM3 prompts), (3) visual PPR seeds.

- **(3) is not new.** MG²-RAG fuses visual and text seed scores before PPR (MG² Eq. 3; Table 7 λ_t/λ_v). EviProp seeds PPR with visual page priors.
- **(1) is not new in the broad sense.** MG²-RAG is graph RAG with native image evidence. HVM-GraphRAG is conflict-resolving "holistic view" graph RAG with images, via captions. What remains new is MemGraphRAG's specific memory (schema filter + conflict resolution) combined with *native* visual nodes. That is an engineering combination, and finding 4 shows the memory contributes little on single-hop KB-VQA.
- **(2) is the only distinctive element, and it is weaker than assumed** (finding 3). The schema vocabulary is seeded with the OntoNotes labels that spaCy also uses. The genuine differences from MG²-RAG are: (a) the *entity set* (LLM triple heads/tails include common nouns such as "impala"; spaCy NER spans mostly do not); (b) the LLM's contextual labelling vs spaCy's labelling errors; (c) whatever off-list types the LLM invents ("not limited to"). None of these were found in any paper. No search turned up type- or ontology-guided prompt selection for segmentation/grounding inside graph RAG (searches: arXiv API phrase queries on multimodal KG RAG, PPR + images, grounding/SAM/GroundingDINO + RAG, ontology/schema + multimodal GraphRAG, HippoRAG + multimodal; web searches for ontology-guided / entity-type-aware grounding in RAG) [V-run searches; absence is not proof].

**Suggested framing that stays defensible:**
"We study *which entities to ground* when building a multimodal graph for RAG. Using a conflict-resolved LLM memory graph (MemGraphRAG), we compare three prompt-selection policies for concept segmentation (NER-label filter as in MG²-RAG; MemGraphRAG schema types; fine-grained LLM types mapped to SAM3 noun phrases). We measure grounding coverage and precision and their effect on retrieval, and compare native visual seeds with entity-aware captioning under leakage controls."
This keeps the system (MemGraphRAG-V) but rests the contribution on a controlled comparison that no prior paper reports. If the type distribution check (finding 3) shows MemGraphRAG emits almost only OntoNotes labels, say so and make the fine-grained variant the "ontology-guided" condition.

---

## 3. Evaluation data

### 3.1 M2KR InfoSeek and E-VQA subsets (Q5)

Source: [M2KR dataset card](https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR) (features from the YAML header), [PreFLMR paper](https://arxiv.org/abs/2402.08327) Table 1 and §3.1, and the parquet files themselves.

| | E-VQA (`EVQA_data` / `EVQA_passages`) | InfoSeek (`Infoseek_data` / `Infoseek_passages`) |
|---|---|---|
| Question fields | `pos_item_ids, pos_item_contents, img_id, img_path, image_id, question_id, question, answers, gold_answer, question_type, instruction` [V-read] | `question_id, image_id, question, answers, answer_eval, data_split, wikidata_value, wikidata_range, entity_id, entity_text, image_path, gold_answer, objects, attribute_scores, attributes, class, ocr, rect, related_item_ids, pos_item_ids, pos_item_contents, ROIs, found, img_caption, instruction, img_path, question_type` [V-read] |
| Test size | 3,750 [V-run] | 4,708 [V-run] |
| Composition | `templated` 1,000 / `automatic` 2,750; **no two-hop, no multi-answer**; 1 gold passage per question; 1,329 distinct gold pages [V-run] | Actually InfoSeek **validation** (`infoseek_val_*`): `val_unseen_entity` 3,841 / `val_unseen_question` 867; types String 3,802 / Time 463 / Numerical 443; 659 distinct entities [V-run]. PreFLMR §3.1: test examples come from the original validation set because InfoSeek test annotations are unreleased [V-read]. |
| Images in rows? | No pixels; `img_path` like `inat/val/…jpg` (2,000) or `google-landmark/train/…` (1,750); 2,350 distinct images [V-run] | No pixels; `image_id` `oven_xxxxxxxx` (all 4,708); 1,722 distinct images [V-run] |
| Passage corpus | 51,472 test passages from 19,267 pages; fields `language, passage_id, passage_content`; **text only** [V-run] | 98,276 passages; fields `passage_id, passage_content, title`; **text only** [V-read card; V-run sample] |
| Licence | MIT (card) | MIT (card) |
| Query-image source | `BByrneLab/M2KR_Images` `EVQA/inat.zip` (8.92 GB) + `EVQA/google-landmark.tar` (2.79 GB); ungated. One image read from the zip via range requests [V-run]. | `M2KR_Images` `Infoseek/infoseek_val_images.tar` (8.96 GB; members named `oven_*.JPEG`, ≈25 KB each) [V-run header peek]. Tars have no index, so stream the file once and keep only the needed members [I]. |

**What the KB-side image source would have to be.**

1. **MuKA URL lists** (COLING 2025; `github.com/lhdeng-gh/MuKA/data_preparation/{evqa,infoseek}_passages_image_urls.jsonl.gz`): 19,408 E-VQA entities with one `image_url` each, keyed by M2KR `passage_id`s, and 34,332 InfoSeek entities keyed by M2KR `title`. URLs are almost all `upload.wikimedia.org` / `commons.wikimedia.org`, plus a few Bing thumbnails. Coverage: **51,472/51,472** M2KR E-VQA test-corpus passages; ≈4,638/4,708 InfoSeek gold entities (by title match) [V-run]. Collected by cascade: Wikipedia main image → first non-trivial page image → Commons search → web search → black placeholder (MuKA §3.2) [V-read]. The repo has no licence file (GitHub API) [V-read].
2. **Official E-VQA KB**: `encyclopedic_kb_wiki.zip` (4.9 GB). Per article: `title, section_titles, section_texts, image_urls, image_reference_descriptions, image_section_indices, url`. The official README points to `TREC-AToMiC/AToMiC-Images-v0.2` for pixels [V-read README]. `image_section_indices` lets images be attached to sections (≈ passages), which is what the method wants.
3. **MG²-RAG's builder** (`examples/data/build_evqa_100k.py`) takes only the **first** `image_urls` entry per KB article and gets pixels by scanning every AToMiC parquet file for matching `image_url`s (lines 49-58, 64-88, 90-140) [V-read]. MG² App. A.2 confirms a random 100k-document sample of the ≈2M-page KB and a 5k subset "ensuring inclusion of the necessary evidence" [V-read].

**Consequence.** With MuKA's or MG²'s data, each KB document has **one image of its main entity**. That is the "visual shortcut" RETINA describes: the query image depicts the target document's main entity. Image→image retrieval then does most of the work, and grounding mostly crops the single depicted entity [I].

### 3.2 E-VQA and InfoSeek facts (Q6)

| | E-VQA ([2306.09224](https://arxiv.org/abs/2306.09224); [repo README](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa)) | InfoSeek ([2302.11713](https://arxiv.org/abs/2302.11713); [repo README](https://github.com/open-vision-language/infoseek)) |
|---|---|---|
| Size | 221k Q+A pairs, 16.7k categories, 1M (I,Q,A) triplets, 514k images (§4.4) | ≈1.3M; train 934,048 / val 73,620 (18,656 unseen-question + 54,964 unseen-entity) / test 347,980; Human set 8,931 (Table 7) |
| Test composition | Test triplets: templated 1,000 / automatic 2,750 / multi-answer 1,000 / two-hop 1,000 = 5,750 (Table 2) | — |
| Two-hop | Chained via bridge entities, generated with PaLM and validated (§4.3) | none |
| Metric | **BEM** via `evaluation_utils.py`, which loads `https://tfhub.dev/google/answer_equivalence/bem/1` with TensorFlow + tensorflow_text | VQA accuracy for STRING and TIME, relaxed accuracy for NUMERICAL; harmonic mean of unseen-question and unseen-entity (§3.3) |
| KB | 2M English Wikipedia articles (WikiWeb2M / WIT, snapshot 13 Aug 2022), with images | 100k Wikipedia KB "(articles and infobox images)" for the With-KB protocol (§4); 6M Wikipedia text dump released |
| Test labels public? | Yes (`test.csv`) | **No**: `infoseek_test.jsonl` rows contain only `data_id, image_id, question` [V-run: first bytes of the file]. KB mapping released for train/val only. EchoSight §3.1 reports on val and notes the original 100k KB was not released, so it filtered a 100k KB from E-VQA's KB. |
| Query images | iNat21 (val split for val/test) and GLDv2 train-clean (§4.4) | OVEN images (`image_id` = `oven_*`) |

### 3.3 Benchmarks with images on the corpus side (Q7)

| Benchmark | Size | Images | Licence | Gold evidence IDs | Hops | Fit |
|---|---|---|---|---|---|---|
| **MultimodalQA (MMQA)**, ICLR 2021, [2104.06039](https://arxiv.org/abs/2104.06039) | 29,918 questions; **dev 2,441**, of which 940 need images and 569 are compositional *and* need images (Compose/Intersect/Compare) [V-run dev file] | 57,058 Wikipedia entity images in `images.jsonl.gz`; `final_dataset_images.zip` **2.36 GB** (S3) [V-run HEAD] | Not stated in repo (GitHub API shows none) | Yes: `supporting_context` doc IDs for train/dev; test has no answers [V-read README] | Single + compositional multi-hop; text questions, images in corpus; also tables | **Best available fit** for H1/H3: text questions whose answer depends on corpus images of entities (e.g. "…{is completely bald and wears thick glasses?}"). Tables must be linearised or those question types skipped. |
| **WebQA**, CVPR 2022, [2109.00590](https://arxiv.org/abs/2109.00590) | train+val 36,766+4,966; test 7,540 [V-read README] | `imgs.tsv` in 51 × 1 GB chunks (≈51 GB) with `imgs.lineidx` [V-read README] | Repo CC0-1.0 [V-read GitHub API] | Yes: `sources` (image/snippet IDs) for train/val [V-read README] | Multi-source (often 2 images) | Good conceptually; the image archive is large. With `imgs.lineidx` only the needed lines need reading once it is downloaded [I]. Test labels: not verified. |
| **CrossModalQA**, arXiv Aug 2026, [2609.05518](https://arxiv.org/abs/2609.05518) | 1,863 QAs; 4,987 articles; 4,431 Commons images; avg 3.5 hops; five path types | Commons | Not stated | Yes, a supporting subgraph per question | Multi-hop cross-modal | Ideal and small, but **no data URL found** (§6). Its baselines are M2RAG and mKG-RAG text variants; no graph MM-RAG baselines. |
| **RETINA**, arXiv Nov 2025, [2511.22843](https://arxiv.org/abs/2511.22843) | 120k train / 2k human-curated test | Related-entity images, avg 4.3 per sample; built on M2KR passages and MuKA image links | ? | Yes | Relational (secondary entity) | Removes the visual shortcut. Release not verified. |
| MRAG-Bench, ICLR 2025, [2410.08182](https://arxiv.org/abs/2410.08182) | 1,353 MC questions, 16,130 images, 9 scenarios | HF `uclanlp/MRAG-Bench`, 4.3 GB (audit evidence) | CC-BY-4.0 [V-read card] | `gt_images` (top-5 ground-truth images) | Single-hop, vision-centric | Image corpus without text documents; poor fit for a text-memory graph. |
| Dyn-VQA, ICLR 2025, [2411.02937](https://arxiv.org/abs/2411.02937) | — | Web search | — | — | Multi-hop, dynamic | Needs live web search; not a fixed corpus. Not a fit. |
| ManyModalQA, AAAI 2020, [2001.08034](https://arxiv.org/abs/2001.08034) | ≈10k (not verified) | — | — | — | Modality disambiguation, single-hop | Not a fit. |
| ViDoSeek (ViDoRAG), EMNLP 2025, [2502.18017](https://arxiv.org/abs/2502.18017) | — | Document pages | — | — | Reasoning over visually rich documents | Document pages, not natural-image KB. Not a fit. |
| MMCoQA | not verified | | | | Conversational | Not checked (§6). |

### 3.4 What MG²-RAG and MemGraphRAG evaluated on (Q8)

| | MG²-RAG ([2604.04969v2](https://arxiv.org/abs/2604.04969)) | MemGraphRAG ([2606.00610v1](https://arxiv.org/abs/2606.00610)) |
|---|---|---|
| Datasets | E-VQA (full test, "5.8k"; Table 3 also reports Single-Hop), InfoSeek (5.8k sampled from the 73k val set), ScienceQA, CrisisMMD (§4.1, App. A.2) | HotpotQA, 2Wiki, MuSiQue (1,000 validation questions each, following HippoRAG), G-Bench Medical and Novel (§5.1, App. F) |
| KB size | 100k random documents (E-VQA; InfoSeek "subset of 100k pages"); 5k subsets with guaranteed evidence when comparing with VaLiK/MMGraphRAG (App. A.2) | The datasets' passage pools; GraphRAG-Bench corpora |
| Metrics | R@K (Table 2), BEM for E-VQA, VQA/relaxed accuracy for InfoSeek (§4.1) | Str-Acc (gold answer contained in output after lower-casing), LLM-Acc (gpt-4o-mini judge), Context Relevance and Evidence Recall (GraphRAG-Bench) (§5.1) |
| Models | spaCy trf, SAM3, EVA-CLIP-8B; per-task tuned hyperparameters (Table 7) | NV-Embed-v2, top-k = 5, gpt-4o-mini for indexing, generation and judging, temperature 0 |
| Caveat | Table 2 (R@1 44.9 on E-VQA) and Table 6 (R@1 57.8 on "E-VQA (5k)") use different KB sizes; Table 2's KB size is not labelled in its caption [V-read] | G-Novel results in Table 1 have no data in the repo |

Comparability: none of our numbers can sit beside these. Use only our own re-runs on identical subsets [I].

### 3.5 Recommendation

1. **Keep a KB-VQA set, but build it correctly.** Use M2KR E-VQA test (sample 300–500 questions). KB = gold pages plus distractor pages from the 19,267-page M2KR E-VQA test corpus. KB images from MuKA URLs (one per page; ~2–5k Wikimedia downloads with a proper User-Agent). Query images from `M2KR_Images` (iNat by range reads; the 2.8 GB GLDv2 tar in full). Label it honestly as a single-hop, entity-recognition-dominated set. Here MG²-RAG and dense CLIP image→image are the baselines that matter. InfoSeek is optional (it adds a 9 GB tar stream and has no extra hop structure).
2. **Add one set where images live in the corpus and questions are text: MMQA dev**, restricted to ImageQ / ImageListQ / TextQ and their compositions if tables are too costly. This is where H1 (visual seeds help multi-hop retrieval), H2 (native vs caption) and H3 (grounding policy) can show something. Check the MMQA licence before use.
3. **If memory and multi-hop are to stay in the story**, add E-VQA two-hop questions from the official `test.csv` with the official KB, using `image_section_indices` to attach images to sections.
4. Use a small dev split (for example 50 MMQA train questions) to set λ; do not tune on test.

---

## 4. Methodological assumptions

### 4.1 MemGraphRAG schema-type granularity (Q9)

- **Plan assumes:** schema types say "what kind of thing" an entity is at a visually useful granularity ("animals, buildings", README).
- **Source says:** the paper defines a type as "an abstract category (e.g., person)" and a schema as "(person, born_in, country)" (§3; App. D: "a high-level taxonomic category (e.g., Person)") [V-read]. The paper's appendix prompts cover only conflict detection and resolution (Figs. 7–8), not schema extraction. The code's schema prompt, `PROMPT['ontology_extraction']` (`code/src/prompts/prompt.py:298-344`, called at `MemGraphRAG.py:381`), reads:
  > "replacing the head and tail entities with their most appropriate entity TYPE … be as specific as possible but concise (one label) … If a type is unclear, return "Unknown". … Common types include but are not limited to: `<CARDINAL>`, `<DATE>`, `<EVENT>`, `<FAC>`, `<GPE>`, `<LANGUAGE>`, `<LAW>`, `<LOC>`, `<MONEY>`, `<NORP>`, `<ORDINAL>`, `<ORG>`, `<PERCENT>`, `<PERSON>`, `<PRODUCT>`, `<QUANTITY>`, `<TIME>`, `<WORK_OF_ART>`"

  These are the 18 OntoNotes 5 labels, the same inventory as spaCy `en_core_web_trf`. The code wraps every returned type in `<…>` (`MemGraphRAG.py:396-401`) [V-read]. Types are extracted **per fact**, so one entity can get several types (`build_memory_graph`, `entity_types[head_type].add(head)`, `:859-860`) [V-read]. A much finer prompt, `PROMPT['entity_type_extraction']` (`prompt.py:2-295`; Species, Building, Bridge, Mountain, Lake, Vehicle, Aircraft, Artwork, FoodDish…), is never referenced anywhere in the code [V-read grep].
- **Implication:** H3 ("ontology-guided beats NER-label prompts") may compare two near-identical label sets. Species (≈half of E-VQA: 2,000/3,750 M2KR test images are iNat) have no OntoNotes label. Whether gpt-4o-mini invents `<Species>`/`<Animal>` or forces them into `<PRODUCT>`/`Unknown` is an empirical question that decides H3 (task P3). Measure it on the smoke-test memory before building the visual layer, and keep the fine-grained-type variant as a planned condition.

### 4.2 SAM3 prompts, licence and availability (Q10)

- **Plan assumes:** SAM3 can be prompted with entities whose type is visual.
- **Source says** ([SAM 3, 2511.16719](https://arxiv.org/abs/2511.16719)) [V-read]:
  - Concepts are restricted "to those defined by simple noun phrases (NPs) consisting of a noun and optional modifiers". Prompts are a short phrase, image exemplars, or both.
  - "SAM 3 struggles to generalize to fine-grained out-of-domain concepts (e.g., aircraft types, medical terms) in a zero-shot manner" (App. B). It is "constrained to simple noun phrase prompts and does not support multi-attribute queries beyond one or two attributes or longer phrases including referring expressions" (App. B).
  - The text encoder "is causal, with a maximum context length of 32" (App. C.2). ≈850M parameters.
  - The SA-Co training data includes fine-grained Wiki-ontology concepts (e.g. "kefir"), so some fine-grained common nouns work. Nothing addresses proper names of people or landmarks.
- MG²-RAG passes normalised entity strings as prompts (`MMGraphRAG.py:582-586` → `sam3Model.py:101-111`) [V-read].
- **Licence** ([SAM License, 19 Nov 2025](https://github.com/facebookresearch/sam3/blob/main/LICENSE)) [V-read]:
  - Non-exclusive, royalty-free right to use, reproduce, distribute and modify. It is not research-only.
  - Redistribution must include a copy of the Agreement (§1.b.i). MG²-RAG's vendored `sam3/` has no licence file (audit), so do not copy it into our public repo without the licence.
  - Publications must acknowledge use of SAM Materials (§1.b.ii).
  - Trade-control and ITAR restrictions apply.
- **Availability:** `facebook/sam3` is `gated: manual`, `library_name: transformers`, files `model.safetensors` and `sam3.pt` [V-run HF API]. Transformers ships `Sam3Model`/`Sam3Processor` with batched text prompts and cached vision embeddings for several prompts per image ([docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3)) [V-read].
- **Implication:** prompting with entity names ("Smilax bona-nox", "Hinrich Lichtenstein") will often return nothing [I]. A type→noun-phrase mapping ("plant", "bird", "building", "bridge", "painting") is what SAM3 is built for. The crop is then linked to the entity that owns the image or passage. That mapping *is* an ontology-guided design and is easy to describe. Fallbacks if access stalls: Grounding DINO or OWLv2 (both in transformers, ungated) [I].

### 4.3 Text→image matching and encoders (Q11)

- **SigLIP2** (`google/siglip2-so400m-patch14-384`): Apache-2.0, ungated, `model.safetensors` 4.54 GB. Its `config.json` sets no `max_position_embeddings`, so the `SiglipTextConfig` default of **64** applies. The transformers SigLIP2 docs say: "we pass `padding=max_length` and `max_length=64` since the model was trained with this", and the processor lower-cases [V-read config, `configuration_siglip.py:52`, `docs/source/en/model_doc/siglip2.md:76,143,163`].
- **EVA-CLIP-8B** (`BAAI/EVA-CLIP-8B`): Apache-2.0, ungated. 8.1B total parameters (card table). HF weights are 4 fp32 `.bin` shards (9.97 + 9.93 + 9.80 + 3.18 = 32.9 GB); the repo also holds a 32.9 GB `.pt` and a 30.1 GB vision-only `.bin` (95.9 GB total). `config.json` has `torch_dtype: float32` and text context 77; custom `modeling_evaclip.py` [V-read]. fp16 weights alone take 8.1e9 × 2 B ≈ **16.2 GB**, so the audit's "~16 GB VRAM" is a lower bound; plan for ≥18–20 GB with activations [I].
- **Evidence on long questions as CLIP queries:** MG² Table 2 (checked on the rendered PDF page). EVA-CLIP-8B T→V R@1/R@10 = 1.8/6.3 on E-VQA and 0.2/0.8 on InfoSeek; T→T 2.2/5.2 and 0.1/1.4; V→V 30.0/50.0 and 47.2/70.2; V→T 42.0/69.5 and 56.5/82.2. MG² sets text fusion weight 0.1 vs visual 1.0 for both datasets (Table 7) [V-read]. The cause here is mostly that questions do not name the entity (finding 4), not only truncation. I found no primary source that isolates "long question vs entity name" as CLIP query [§6].
- **MG² ablations:** there is **no encoder ablation**. The closest thing to an object-level ablation is "w/o MNF" (Multimodal Node Fusion, i.e. aligning text entities with grounded regions): on E-VQA (5k), R@1/R@5 57.8/83.1 → 43.8/78.0 and BEM 60.24 → 55.38. Removing graph propagation drops R@1 to 25.5. Removing the hierarchy gives 40.7/61.5 (Table 6). Sensitivity: varying λ_v from 0.5 to 1.5 changes E-VQA R@1 only between 56.7 and 57.8 (Table 8) [V-read]. A λ sweep may therefore show little.
- **Implication:** SigLIP2 vs EVA-CLIP-8B is an uncontrolled change relative to MG²-RAG. Either run MG²-RAG with SigLIP2 as well, or report the MG² comparison with EVA-CLIP-8B. For joint image+question queries, `Qwen/Qwen3-VL-Embedding-2B` (Apache-2.0, 4.26 GB, 32k context, text/image inputs, [2601.04720](https://arxiv.org/abs/2601.04720)) avoids the 64/77-token limits [V-read card].

### 4.4 Caption baseline strength (Q12)

- **Plan assumes:** "caption every image with Qwen2.5-VL and index captions as text" is the main baseline.
- **Sources say:**
  - On documents, VisRAG reports a 20–40% end-to-end gain over text-based (parsing) RAG ([2410.10594](https://arxiv.org/abs/2410.10594) abstract).
  - In MG²'s own setting, caption-style graph baselines are very weak: VaLiK 15.22 and MMGraphRAG 16.52 BEM on E-VQA 5k (All) vs MG²-RAG 53.36, all with Qwen2.5-VL-7B (Table 3).
  - Entity recognition of the query image followed by text retrieval is strong: E-VQA Lens → KB section gives PaLM 48.8% vs 87.0% oracle (Tab. 5). TrioRAG finds the VLM-generated query is the robust signal.
  - MRAG-Bench (card) reports that LVLMs gain more from images than from textual knowledge in vision-centric scenarios.
  - The controlled 2026 study (2607.16604) finds multimodal gains sensitive to textual leakage [V-read abstracts/tables].
- **Implication:** the outcome of H2 depends on the caption baseline's design. A fair, strong version:
  - Caption the query image too.
  - Use an entity-aware prompt ("name the most specific entity shown — species, landmark, artwork — then describe").
  - Use the same MLLM for captions and answers in every system.
  - Index captions as passages linked to their page.
  - Report both a generic and an entity-aware variant.
  - Decide explicitly whether Wikipedia image captions (`image_reference_descriptions` in the E-VQA KB) are allowed; they leak entity names.

### 4.5 Model sizes and licences; newer captioners (Q13)

| Model | Licence | Gated | Weights | Notes |
|---|---|---|---|---|
| `BAAI/EVA-CLIP-8B` | Apache-2.0 | no | 32.9 GB (HF fp32 shards) | §4.3 |
| `google/siglip2-so400m-patch14-384` | Apache-2.0 | no | 4.54 GB | 64-token text |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 | no | 16.58 GB | used in MG² Table 3 |
| `Qwen/Qwen3-VL-8B-Instruct` | Apache-2.0 | no | 17.53 GB | created 11 Oct 2025; [tech report 2511.21631](https://arxiv.org/abs/2511.21631) |
| `Qwen/Qwen3-VL-4B-Instruct` | Apache-2.0 | no | 8.88 GB | fits the MIG slice more easily [I] |
| `Qwen/Qwen3.5-9B` / `-4B` | Apache-2.0 | no | 19.31 / 9.32 GB | `image-text-to-text` pipeline; MG² uses Qwen3.5-27B as its strongest open MLLM |
| `Qwen/Qwen3-VL-Embedding-2B` | Apache-2.0 | no | 4.26 GB | joint text/image embedding |

All sizes from the HF API (sum of `.safetensors`) [V-run]. Recommendation: Qwen3-VL-8B (or Qwen3.5-9B) as captioner and generator. Keep Qwen2.5-VL-7B only if matching MG²'s Table 3 row matters more than strength [I].

### 4.6 LLM cost (Q14)

- OpenAI pricing page ([developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing), page data read 3 Oct 2026) [V-read]:
  - **gpt-4o-mini:** $0.15 input / $0.075 cached / $0.60 output per 1M (Standard); $0.075 / $0.30 (Batch).
  - gpt-5-nano $0.05 / $0.40; gpt-4.1-mini $0.40 / $1.60; `gpt-6-luna` $0.10 / $0.01 cached / $0.50 output (Batch $0.05 / $0.25).
- Deprecations page ([developers.openai.com/api/docs/deprecations](https://developers.openai.com/api/docs/deprecations)) [V-read]:
  - gpt-4o-mini appears only as the recommended *replacement* in old fine-tuning entries. It is **not scheduled for shutdown**.
  - `gpt-4.1-nano` shuts down **23 Oct 2026**, the deadline day.
  - `gpt-5-nano-2025-08-07` shuts down 11 Dec 2026.
- `docs/audit/evidence/indexing_cost_estimate.json` uses $0.15 / $0.60 (`scripts/estimate_indexing_cost.py:23`) [V-read]. The $13–37 per corpus range still holds. The Batch API halves it, but MemGraphRAG calls the LLM synchronously, so batching needs a wrapper [I]. The fixed prompt parts (218–695 tokens) are below OpenAI's 1,024-token caching threshold, so expect no cache discount [I; threshold not re-verified].

### 4.7 BGE pooling and query instruction (Q15)

- **Model card** ([BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)) [V-read]:
  - Sentence-transformers pooling config `1_Pooling/config.json` has `"pooling_mode_cls_token": true`.
  - The query instruction for retrieval is "Represent this sentence for searching relevant passages: ". For v1.5, "no instruction only has a slight degradation", and it is recommended for short queries.
- **MemGraphRAG `src/embedding_model/BGE.py`** [V-read]:
  - Uses `mean_pooling` (`:17`, `:112`); the comment at `:16` wrongly claims the official BGE examples use mean pooling.
  - Sets `"instruction": ""` (`:90`).
  - `batch_encode` merges an `instruction` kwarg into `params` (`:116-123`), but `encode()` never reads it (`:99-113`), so the HippoRAG query instructions passed from `MemGraphRAG.py:1964-2039` are silently dropped.
- **Implication:** the audit's claim is confirmed; upgrade it from [I] to [V-read]. Keep the wrapper as released for "MemGraphRAG (released code)", or fix it in all systems equally, and say which.

### 4.8 HippoRAG 2 vs "MemGraphRAG as released" (Q16)

- **HippoRAG 2** ([2502.14802](https://arxiv.org/abs/2502.14802)) [V-read]:
  - Phrase (entity) seeds come from triples that pass query-to-triple retrieval and the recognition-memory LLM filter. Up to k phrase nodes are chosen by "their average ranking scores across filtered triples".
  - "All passage nodes are also taken as seed nodes", with similarity-proportional reset scaled by a weight factor set to **0.05** (§3.5, §6.2, Table 5).
  - PPR damping **0.5** (appendix hyperparameter table).
  - Dense fallback when no triples survive (§3.5).
- **MemGraphRAG code:** same seed rule family (audit: `MemGraphRAG.py:2124-2166`), passage factor 0.05, damping 0.5, dense fallback (`:1100-1105`), recognition filter optional (DSPy) or a cosine threshold. Type nodes get zero reset mass but still take part in the walk through entity–type and type–type edges [V-read audit; I].
- **Implication:**
  - The released retrieval is HippoRAG 2's retrieval over a different graph: schema-filtered, conflict-resolved facts plus type nodes.
  - MemGraphRAG's legacy `index()` path is effectively HippoRAG 2 on an OpenIE graph inside the same codebase. Name it "w/o memory (≈HippoRAG 2 pipeline)", and run it with the recognition filter set as in HippoRAG 2 if you claim equivalence.
  - That gives a HippoRAG 2 baseline at no extra cost. Running the official HippoRAG 2 repo is not necessary [I].
  - In the paper, describe the text system as "MemGraphRAG (released code; paper's Eqs. 7–8 not implemented)".

---

## 5. Other issues (Q17)

1. **Dispatch problem is the common case, not an edge case.** On E-VQA/InfoSeek almost every question is deictic (finding 4). MemGraphRAG's retrieval will find no fact above threshold and fall back to dense passage retrieval (`MemGraphRAG.py:1100-1105`) before any visual seed can act. The visual channel must be wired in before that check. P4 in task 03 should be marked blocking.
2. **H1 design.** Compare against dense CLIP image→image/text retrieval and MG²-RAG, not only text-only MemGraphRAG. A text-only system cannot know which plant "this plant" is.
3. **One image per KB document weakens H3 on KB-VQA.** Crops of the single depicted entity add little over the whole image [I]. H3 needs scenes with several entities (MMQA images, WebQA, E-VQA KB articles with several section images).
4. **Leakage control for H2.** Entity-aware captions of the query image essentially solve recognition. Report retrieval with and without captions on the query side so readers can see where the gain comes from (2607.16604).
5. **Metrics.** E-VQA BEM requires TensorFlow + tensorflow_text + TF Hub (`evaluation_utils.py:25-31, 305`). If that is a problem on Snellius, use containment + LLM judge and say so. For InfoSeek, implement VQA/relaxed accuracy per question type.
6. **Gold mapping for the text regression is easier than the audit says.** The repo ships `dataset/*/*_corpus.json` with per-passage `title`/`text` (e.g. 9,811 HotpotQA passages), and the question files keep `supporting_facts` titles [V-run]. Indexing one passage per chunk keeps gold IDs.
7. **Citations.** MG²-RAG: cite ECCV 2026. MemGraphRAG: KDD 2026 (DOI above). Acknowledge SAM 3 per its licence.
8. **Timeline.** The 2 Oct gate has passed; this review did not check its outcome. The data changes above (MuKA images, M2KR_Images, MMQA) take ≈1–2 days and should replace "fetch only the needed images" in the 5–9 Oct loader slot [I].

---

## 6. Unverified / could not access

- **Google Scholar forward citations** were not checked; Semantic Scholar may lag by weeks.
- **CrossModalQA and RETINA data releases**: no URL found in the papers (via WebFetch summaries) or by web search.
- **Licences**: MMQA's licence is not stated in its repo. MuKA's repo has no licence file. Wikimedia image licences vary per image.
- **WebQA test-label availability** and its image-only subset size were not verified.
- **MMCoQA, M2RAG (2025) and ViDoSeek sizes** were not checked.
- **MMGraphRAG venue** (v3, Jul 2026) not verified. LILaC's EMNLP 2025 venue is taken from its project URL only.
- **No primary source isolates "long question vs entity name" as CLIP query.** The claim in §4.3 rests on MG² Table 2, where questions do not name the entity.
- **VRAM figures** (EVA-CLIP-8B ≥16.2 GB fp16, SAM3) are arithmetic or inferred, not measured.
- **OpenAI prompt-caching threshold** (1,024 tokens) was not re-verified.
- **Paper details read through summaries**: the descriptions of mKG-RAG, MMGraphRAG, RAG-Anything, HVM-GraphRAG and CEMMKG come from WebFetch summaries of their arXiv HTML. Re-read the method sections before citing specifics.
- **InfoSeek val tar coverage**: whether `infoseek_val_images.tar` holds all 1,722 images needed by the M2KR InfoSeek test is inferred from member naming, not checked exhaustively.
- **MemGraphRAG's actual type output** (the decisive check for H3) cannot be known without a real run on gpt-4o-mini.
