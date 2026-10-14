# MemGraphRAG-V: methodology

## 1. Research question and scope

**Does graph-guided retrieval of native image evidence improve answers to text questions compared with caption-based MemGraphRAG and direct text-to-image retrieval?**

The system receives a text question, retrieves relevant corpus images and associated text, and passes them to a vision-language model (VLM).

We investigate three mechanisms:

1. **Visual graph connections:** SAM3 links depicted objects to MemGraphRAG entities.
2. **Visual retrieval seeds:** question–image and question–crop similarities guide graph retrieval.
3. **Native evidence:** the reader model inspects original images instead of generated descriptions.

MemGraphRAG supplies the text memory and graph. MG² supplies the inspiration for visual grounding, representations and retrieval seeds. MG² already supports text-to-image matching; our contribution concerns its integration with MemGraphRAG and the explicit selection of native image evidence.

**Intended method:** whole-image nodes, SAM3 grounding, SigLIP2 image/crop embeddings, CPU Personalized PageRank (PPR), and a multimodal reader VLM.

SAM3 only runs during indexing. Question-time retrieval uses its saved outputs.

Visual fact extraction and multimodal conflict resolution remain outside scope.

## 2. Dataset and shared configuration

**Benchmark selection: Misha.**

Select a benchmark providing:

- Text questions understandable without a query image.
- Corpus images containing evidence needed to answer.
- Gold answers and supporting document IDs.
- An official answer-scoring implementation.
- Multiple candidate documents, including distractors.

Prefer image-level evidence annotations. Do not remove query images from questions that depend on them.

**Provisional size target:** 100 development and 400 held-out test questions, subject to benchmark availability. Use official splits where possible. Otherwise save a grouped split with random seed `42`, keeping questions sharing supporting documents together.

Every system uses the same corpus, questions and evidence mappings. Include a fixed distractor set. Answers and gold evidence labels must not enter indexing or captioning prompts.

Create these manifests:

| Manifest | Required fields |
|---|---|
| Text | `doc_id`, `passage_id`, text, source location |
| Images | `image_id`, `doc_id`, associated passage IDs, path, page/figure |
| Crops | `crop_id`, `image_id`, existing entity ID, bounding box, confidence, path |
| Questions | Question ID/text, answers, gold document IDs, split; gold image IDs where available |

Validate missing files and passage–image mappings before indexing. Report corpus coverage against the original gold labels.

The benchmark and answering VLM remain unselected. Before experiments, record their exact versions, official scorer, reader image settings and generation settings. Use the same reader throughout, with deterministic decoding where supported.

## 3. Index construction

### 3.1 Build MemGraphRAG’s text memory

Run: Fact extraction → schema extraction → ontology filtering → conflict detection/resolution → final memory graph

Pin the upstream commit and configuration. **Preserve an unaugmented graph for the text baseline comparison.**

Reuse the final entity and passage IDs. **We do not build a separate MG² graph or independently create entities and merge them afterward.**

Any necessary correctness fixes must be documented and applied consistently across applicable MemGraphRAG configurations.

### 3.2 Attach whole images

Add one node per corpus image. Connect it to passages that explicitly contain or reference it.

Assign each image–passage edge weight `1.0`; deduplicate repeated associations. Preserve original text edges and weights.

Retain original captions and surrounding text. Generated image descriptions are reserved for caption experiments.

An image remains retrievable even when SAM3 finds no object: it still has an image embedding and passage connections.

### 3.3 Ground existing entities with SAM3

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

### 3.4 Store crops without adding crop vertices

Each crop is a saved visual record linked to its parent image and existing entity. It is not a separate PPR vertex.

Reuse MG²’s crop-generation approach and record its exact preprocessing, including masking and background treatment. Freeze this implementation before experiments.

Keep the complete parent image. **Crops support retrieval; the reader receives whole images**, preserving axes, legends and diagram structure.

## 4. Visual encoder and execution environment

Use **`google/siglip2-so400m-patch14-384`** as the single visual encoder throughout the study. Pin model and processor revisions.

Implement one adapter:

```text
encode_images(images_or_crops) → normalized vectors
encode_text(questions)         → normalized vectors
```

Use the checkpoint’s processor and projected features. L2-normalize vectors; their dot product gives cosine similarity. Encode questions directly without query expansion.

Precompute and cache all whole-image and crop embeddings. Cache query embeddings as well. Cache keys include model/processor revision and an input-content hash.

Record failed images and truncated questions. Never silently substitute blank images.

Use identical whole-image embeddings in the direct-retrieval baseline and every graph configuration. Keep MemGraphRAG’s text encoder separate; never compare its vectors directly with SigLIP2 vectors.

Retain MemGraphRAG’s **CPU PPR**. If dependencies conflict, run SAM3 and SigLIP2 preprocessing in separate environments and export embeddings and ID manifests.

Freeze model revisions, preprocessing and the adapter before test evaluation.

## 5. Retrieval and answer generation

### 5.1 Compute both channels before choosing a route

For every text question, independently compute:

- **Text seeds:** MemGraphRAG’s existing fact/passage seed weights.
- **Image candidates:** top 20 whole images by SigLIP2 similarity.
- **Crop candidates:** top 20 crops by SigLIP2 similarity.

Use all available items when an index has fewer than 20. Break ties by stable ID.

Visual retrieval must run even when no textual facts survive. The routing must be independent.

Reject nonfinite scores and clamp negative weights to zero. A component is available when its retained weights have a positive sum.

### 5.2 Convert visual matches into seeds

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

### 5.3 Run PPR and select evidence

Run weighted, undirected CPU PPR:

```text
p = αs + (1 − α)Pᵀp
```

`α` is the restart probability, `s` the seed distribution, `P` the transition matrix and `p` the final node scores.

For image-based retrieval:

1. Read final image-node scores.
2. Give each document the maximum score of its images.
3. Return the top **five unique documents**.
4. Select the highest-scoring image within each document.

The reader therefore receives at most five images, one per document. This fixed limit may miss evidence requiring multiple figures within one document; report that limitation.

For passage-based systems, rank passages, deduplicate by document and retain the highest-scoring passage per document.

### 5.4 Supply native evidence to the reader

For each selected image, provide:

- The actual whole image.
- A source identifier.
- Up to 256 tokens of original caption and associated text, assembled in source order.

Use the same text-attachment policy for direct and graph-based image retrieval. Keep reader settings and answer instructions fixed.

Save selected evidence IDs, scores, retrieval route, prompts and answers for every question.

## 6. Experiments

### Main comparisons

| ID | System | Reader evidence | Purpose |
|---|---|---|---|
| T | Original text MemGraphRAG | Retrieved text | Text baseline |
| C | Caption-based MemGraphRAG | Text including generated descriptions | End-to-end baseline |
| D | Direct SigLIP2 image retrieval | Whole images and associated text | Test graph retrieval |
| G | Full MemGraphRAG-V | Whole images and associated text | Intended method |
| G-caption | G’s saved evidence selection | Descriptions replacing images | Test evidence representation |

For C, use the chosen reader VLM to generate one description per image with a fixed, question-independent prompt and a 256-token output limit. Request visible labels, values, trends and relationships without guessing. Cache descriptions and index them with the text.

Caption conditions receive up to 256 tokens of original associated text plus 256 tokens of generated description per document. Native conditions receive the same associated text and the image instead of its description. Report their actual costs.

For **G-caption**, reuse G’s exact selected image IDs, document order and associated text. Do not retrieve again.

Interpretation:

- **G versus D:** benefit of the full graph-guided selector.
- **G versus G-caption:** benefit of native images relative to the chosen description procedure.
- **G versus C:** overall pipeline difference; it cannot independently identify the cause of a possible performance improvement.

### Component ablations

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

## 7. Evaluation

### Primary retrieval metric: Document-ID Recall@5

For question `q`, let `R₅(q)` be the five returned document IDs and `G(q)` its gold supporting document IDs:

```text
Recall@5(q) = |R₅(q) ∩ G(q)| / |G(q)|
```

Report the mean across questions. Use benchmark conventions for alternative valid evidence and require nonempty gold evidence sets.

Where image annotations exist, report recall over the **images actually supplied to the reader**. A correct document ID does not guarantee that the selected figure contains the needed evidence. Otherwise assess this through manual inspection.

### Primary QA metric: official benchmark score

Use the benchmark authors’ scorer and prescribed normalization; record its version.

**Containment accuracy** and a **fixed LLM judge** are supplementary but necessary. Freeze the judge model and rubric, hide system identities and cache decisions.

### Uncertainty

Report 95% bootstrap confidence intervals using 2,000 resamples and seed `42`.

Resample dependent question groups together when they share supporting documents or benchmark-defined dependencies; otherwise resample questions. Report the grouping rule and number of groups.

For system differences, use identical sampled groups in each replicate. Report paired differences for the main comparisons and SAM3 ablations.

These intervals represent evaluation-sample uncertainty, not variability from rebuilding graphs with stochastic models.

### Error analysis

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

### Cost and text regression

Report indexing time, API tokens, SAM3 preprocessing time, embedding time, crop count and index size. Report median and 95th-percentile retrieval latency on the same hardware, excluding generation. Report generation cost separately.

Run T and the extension on one fixed text-dataset subset without images. Report score differences; a nonsignificant difference alone does not establish no degradation.

## 8. Execution and completion criteria

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