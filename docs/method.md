# MemGraphRAG-V: methodology

**Current scope.** The study compares caption-augmented MemGraphRAG with MemGraphRAG-V on a prepared WebQA subset. Both systems index original WebQA text and captions together with the same generated image descriptions. MemGraphRAG-V additionally represents corpus images in the graph, uses SAM3 grounding and EVA-CLIP image/crop embeddings, and supplies retrieved whole images to a multimodal reader. The combined pipeline implements indexing and retrieval; the WebQA runner implements orchestration, answer generation and result collection. Metric computation is performed separately.

## 1. Research question and scope

**Does MemGraphRAG-V, using native image evidence in addition to caption-augmented text, improve document retrieval and question answering over caption-augmented MemGraphRAG?**

The system receives a text question, retrieves relevant corpus images and associated text, and passes them to a vision-language model (VLM).

The extension combines three mechanisms:

1. **Visual graph connections:** SAM3 links depicted objects to MemGraphRAG entities.
2. **Visual retrieval seeds:** question–image and question–crop similarities guide graph retrieval.
3. **Native evidence:** the reader model inspects original images and generated descriptions.

MemGraphRAG supplies the text memory and graph. MG² supplies the inspiration for visual grounding, representations and retrieval seeds. MG² already supports text-to-image matching; our contribution concerns its integration with MemGraphRAG and the explicit selection of native image evidence.

**Intended method:** whole-image nodes, SAM3 grounding, `EVA-CLIP-8B` image/crop embeddings, CPU Personalized PageRank (PPR), and a multimodal reader VLM.

SAM3 only runs during indexing. Question-time retrieval uses its saved outputs.

Visual fact extraction and multimodal conflict resolution remain outside scope.

## 2. Dataset and shared configuration

**Benchmark selection: Misha.**


> We use a prepared subset of WebQA containing 480 development questions and 120 test questions. The corpus contains 7,859 original text snippets, 4,221 original image-caption passages, 4,221 images and 4,221 additional generated image descriptions. Both systems index the same 16,301 text records before chunking and use the same question sets, supporting-document mappings and distractor corpus. MemGraphRAG-V additionally indexes the images and detected crops as visual evidence.
>
> The supplied `build_qa.json` is used as the development set and `test_qa.json` as the test set. This is a custom WebQA subset, not a claim to reproduce the full official benchmark. All 120 supplied test questions have image-only gold evidence. The split has overlapping supporting sources and is not evidence-disjoint. Questions and reference answers are not passed to the indexing, caption-generation or reader prompts as labels; reference labels are retained separately for evaluation.
>
> Additional descriptions were prepared using Qwen/Qwen2.5-VL-7B-Instruct with a question-independent visual-description prompt. The supplied generation script requests three to five factual sentences and sets `max_new_tokens=150`. Both systems reuse the same saved descriptions. 

| Manifest | Required fields |
|---|---|
| Text | `doc_id`, `passage_id`, text, source location |
| Images | `image_id`, `doc_id`, associated passage IDs, path, page/figure |
| Crops | `crop_id`, `image_id`, existing entity ID, bounding box, confidence, path |
| Questions | Question ID/text, answers, gold document IDs, split; gold image IDs where available |

## 3. Model roles

Role | Model |
|---|---|
| Text extraction and memory construction | Qwen/Qwen2.5-7B-Instruct, served locally through vLLM |
| Text embeddings | BAAI/bge-large-en-v1.5 |
| Image, crop and visual-query embeddings | BAAI/EVA-CLIP-8B |
| Generated descriptions | Qwen/Qwen2.5-VL-7B-Instruct |
| MemGraphRAG-V answer reader | Qwen/Qwen2.5-VL-7B-Instruct, loaded through Transformers |

## 4. Index construction

### 4.1 Build MemGraphRAG’s text memory

Run: Fact extraction → schema extraction → ontology filtering → conflict detection/resolution → final memory graph

Pin the upstream commit and configuration. **The baseline graph is built from original WebQA text and captions plus the supplied generated descriptions. It contains no image vertices or SAM3 image–entity edges. The extension uses the same text inputs and text-processing settings before adding visual structure.**

Reuse the final entity and passage IDs. **We do not build a separate MG² graph or independently create entities and merge them afterward.**

Any necessary correctness fixes must be documented and applied consistently across applicable MemGraphRAG configurations.

### 4.2 Attach whole images

Add one node per corpus image. Connect it to passages that explicitly contain or reference it.

Assign each image–passage edge weight `1.0`; deduplicate repeated associations. Preserve original text edges and weights.

Retain original captions and surrounding text. Original captions and generated descriptions are included in the textual index for both systems. In MemGraphRAG-V, image–passage associations also connect each image to its generated-description record.

An image remains retrievable even when SAM3 finds no object: it still has an image embedding and passage connections.

### 4.3 Ground existing entities with SAM3

For each image:

1. Collect final MemGraphRAG entities linked to its associated passages and caption.
2. Deduplicate candidate entity IDs.
3. Submit each entity’s name as a SAM3 text prompt on that image.
4. Retain detections with confidence at least `0.5`.
5. Save bounding boxes, masks, crops, confidence and the original entity IDs.
6. Add an image–entity edge when at least one detection survives.

Use the maximum retained grounding confidence as the edge weight for each image–entity pair. Multiple detections must not multiply the edge weight.

**SAM3 locates prompted entities; it does not create new textual entities or verify facts.** A detection is a model prediction that may be wrong.

In the core method, textual co-occurrence supplies grounding candidates but does not directly create image–entity edges. This keeps the meaning of those edges clear and allows us to measure what SAM3 adds.

### 4.4 Store crops without adding crop vertices

Each crop is a saved visual record linked to its parent image and existing entity. It is not a separate PPR vertex.

Reuse MG²’s crop-generation approach and record its exact preprocessing, including masking and background treatment. Freeze this implementation before experiments.

Keep the complete parent image. **Crops support retrieval; the reader receives whole images**, preserving axes, legends and diagram structure.

## 5. Visual encoder and execution environment

Use **``BAAI/EVA-CLIP-8B``** as the single visual encoder throughout the study. Pin model and processor revisions.

Implement one adapter:

```text
encode_images(images_or_crops) → normalized vectors
encode_text(questions)         → normalized vectors
```

Use **``openai/clip-vit-large-patch14``** and projected features. L2-normalize vectors; their dot product gives cosine similarity. Encode questions directly without query expansion.

Precompute and cache all whole-image and crop embeddings. Cache query embeddings as well. Cache keys include model/processor revision and an input-content hash.

Record failed images and truncated questions. Never silently substitute blank images.

## 6. Retrieval and answer generation

### 6.1 Compute both channels before choosing a route

For every text question, independently compute:

- **Text seeds:** MemGraphRAG’s existing fact/passage seed weights.
- **Image candidates:** top 20 whole images by EVA-CLIP similarity.
- **Crop candidates:** top 20 crops by EVA-CLIP similarity.

Use all available items when an index has fewer than 20. Break ties by stable ID.

Visual retrieval must run even when no textual facts survive. The routing must be independent.

Reject nonfinite scores and clamp negative weights to zero. A component is available when its retained weights have a positive sum.

### 6.2 Convert visual matches into seeds

Image similarity supplies weights to image nodes.

Crop similarity supplies weights to existing entity nodes through saved crop–entity mappings. For repeated detections, take the maximum similarity within each image–entity pair, then the maximum across images for each entity.

This avoids additive rewards for producing many crops. Keep query similarity separate from SAM3 confidence: confidence weights graph edges; similarity weights query seeds.

Normalize image and crop-derived entity components independently. When both are available:

```text
ŝ_visual = 0.5 ŝ_image + 0.5 ŝ_crop-entity
```

Otherwise use the available component alone.

Normalize text seeds independently. When both text and visual channels are available:

```text
s = (1 − β) ŝ_text + β ŝ_visual
```

If only one channel is available, assign it all restart mass.

Tune `β ∈ {0.25, 0.50, 0.75}` on development-set Document-ID Recall@5 for the full method. Break ties in favour of the smaller value. Freeze β for the test set and component ablations.

If neither channel is available, use the shared dense-passage fallback. For native-image conditions, attach the first associated image in source order to each selected fallback passage, where available. Log every fallback.

### 6.3 Run PPR and select evidence

Run weighted, undirected CPU PPR:

```text
p = αs + (1 − α)Pᵀp
```

`α` is the restart probability, `s` the seed distribution, `P` the transition matrix and `p` the final node scores.

For graph-based retrieval:

1. Read the final PPR scores of passage and usable image nodes.
2. Score each document by the maximum PPR score among its passage and
   image nodes.
3. Select up to five distinct documents with positive scores, ranked by
   descending document score. Break equal document scores by document ID.
4. For each selected document, retain its highest positive-scoring passage
   chunk's text record and its highest positive-scoring usable image,
   when available. Break equal chunk or image scores by their IDs.
5. Add the selected image's associated text records, removing duplicates.

A document can be selected through its passage scores even when it has no
image. The reader receives at most five original images, one per selected
document. Passage and image scores come from the same PPR distribution;
no additional modality-specific score normalization is applied here.

If there are no usable seeds, or no passage/image nodes receive positive
PPR scores, use the existing dense-passage fallback. The fallback ranks
chunks by query similarity, keeps the highest-scoring chunk per document,
and returns up to five documents with their associated evidence.

### 6.4 Supply native evidence to the reader

For each selected document, provide:

- A source identifier.
- Its selected original whole image, when available.
- Up to 256 tokens of original caption and passage text.

For graph-selected documents, place the best passage chunk's text record
first, followed by the selected image's associated text records, removing
duplicates. The reader excludes generated descriptions, preserves the
remaining record order, and applies the text budget after concatenation.
Selecting a chunk identifies its source record; the current reader does
not extract a separate excerpt centered on that chunk.

The dense fallback retains its existing evidence-attachment policy.
Keep reader settings and answer instructions fixed across comparisons.

Save selected evidence IDs, passage/image PPR diagnostics, document scores,
retrieval route, prompts and answers for every question.

## 7. Experiments

### Main comparisons

We plan two main runs on the same prepared WebQA corpus and question sets. Both systems receive the same original text, original captions and generated descriptions during indexing. Only MemGraphRAG-V adds image vertices, SAM3-derived image–entity edges and visual retrieval seeds.

| ID | System | Indexed evidence | Reader evidence |
|---|---|---|---|
| C | Caption-augmented MemGraphRAG | Original text and captions plus generated descriptions; textual graph only | Retrieved textual evidence, including captions/descriptions |
| G-augmented | MemGraphRAG-V | The same textual corpus plus whole-image vertices, SAM3 grounding and EVA image/crop embeddings | Retrieved whole images and original associated text |

> C is a caption-augmented application of MemGraphRAG, rather than a reproduction of its original paper's benchmark results. G-augmented is the name used by the implemented runner. We compare document retrieval and final answer quality to assess the overall effect of adding the multimodal extension. Because retrieval structure, seeding, evidence selection and reader input modality change together, this comparison cannot isolate their individual contributions.
>
> Direct image retrieval, an uncaptioned text baseline, G-caption and the G-image/G-ground/G0 component ablations are not included in the current experiment plan. The limitation is that an observed difference cannot be attributed specifically to SAM3, crop seeds or image reading.

### 7.1 Component ablations

| Configuration | Image–passage edges | SAM3 image–entity edges | Visual seeds |
|---|---|---|---|
| G-image | Yes | No | Whole images |
| G-ground | Yes | Yes | Whole images |
| G: full method | Yes | Yes | Whole images and crops |
| G0 | Yes | Yes | None |

These comparisons answer distinct questions:

- **G-ground versus G-image:** do SAM3-supported graph connections help?
- **G versus G-ground:** do crop-derived entity seeds help?
- **G versus G0:** does visual seeding help on the same graph?

Keep all other settings fixed. In G0, missing text seeds trigger the fallback; they must not reactivate visual seeds.

For a separate graph-augmentation diagnostic, compare T with G0 using the same passage-score-to-document ranking. Do not mix this comparison with a change to image-based output selection.

The no-SAM3 version is an implementation milestone and control, not the final intended method.

## 8. Evaluation

### 8.1 Primary retrieval metric: Document-ID Recall@5

For question `q`, let `R₅(q)` be the five returned document IDs and `G(q)` its gold supporting document IDs:

```text
Recall@5(q) = |R₅(q) ∩ G(q)| / |G(q)|
```

Report the mean across questions. Use benchmark conventions for alternative valid evidence and require nonempty gold evidence sets.

Where image annotations exist, report recall over the **images actually supplied to the reader**. A correct document ID does not guarantee that the selected figure contains the needed evidence. Otherwise assess this through manual inspection.

### 8.2 Primary QA metric: official benchmark score

Use the benchmark authors’ scorer and prescribed normalization; record its version.

**Containment accuracy** and a **fixed LLM judge** are supplementary but necessary. Freeze the judge model and rubric, hide system identities and cache decisions.

### 8.3 Error analysis

Inspect up to 30 failed G answers sampled with seed `42`, plus up to 10 cases where C or D succeeds and G fails. Deduplicate overlaps.

Two teammates independently classify cases, then resolve disagreements. Allow multiple labels:

| Failure | What to inspect |
|---|---|
| Visual encoder | Correct image/crop has low similarity |
| Graph propagation | Relevant seeds lose rank after PPR |
| Passage–image mapping | Figure is linked to incorrect text |
| SAM3 grounding | Wrong object detected or relevant object missed |
| Candidate entity coverage | Required entity was never available to prompt SAM3 |
| Reader | Correct evidence supplied but misinterpreted |
| Missing evidence | Required evidence absent from corpus or reader bundle |
| Caption advantage | Description exposes a fact the image reader misses |
| Unclear/other | Insufficient evidence for diagnosis |

Inspect candidate entities, detections, direct ranks, seeds, graph scores and actual reader inputs before assigning causes.

### 8.4 Cost and text regression

Report indexing time, API tokens, SAM3 preprocessing time, embedding time, crop count and index size. Report median and 95th-percentile retrieval latency on the same hardware, excluding generation. Report generation cost separately.

Run T and the extension on one fixed text-dataset subset without images. Report score differences; a nonsignificant difference alone does not establish no degradation.

## 9. Execution and completion criteria

| Stage | Required output |
|---|---|
| Data/setup | Misha’s benchmark selection, chosen reader, validated manifests, real MemGraphRAG smoke run |
| Whole-image milestone | Cached image embeddings, image–passage graph, direct retrieval and working reader |
| SAM3 integration | Validated detections, image–entity edges, crop mappings and cached crop embeddings |
| Full retrieval | Independent channels, calibrated seeds, CPU PPR and five-document output |
| Evaluation | Main comparisons, component ablations, uncertainty, costs and error analysis |
| Reporting | Frozen configuration, reproducible artifacts and evidence-supported conclusions |

Target complete implementation by **13 October**, main results by **16 October**, and submission on **23 October**. Write the method during implementation.

Validate SAM3 on a small sample early, including failed detections, before processing the full corpus. A grounding failure for an individual image must not remove that image from whole-image retrieval.

If time becomes tight, omit ontology-filtering experiments and additional model variants. Retain SAM3 in the intended method. If it cannot be integrated reliably, explicitly report the reduced implementation and revise the contribution accordingly.

Save one experiment configuration containing model revisions, upstream commits, prompts, preprocessing, corpus/split hashes, graph settings, thresholds, seeds and scorer versions. Begin the main test evaluation only after this configuration is frozen.