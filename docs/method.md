# MemGraphRAG-V: methodology proposal

This document describes what we will build, what we will test, and how we will test it. It is written for readers who are new to graph-based retrieval. Section 2 explains the background you need, and a glossary is at the end. The paper is due on **23 Oct 2026** (9 pages, ACM format).

---

## 1. The idea in one paragraph

MemGraphRAG is a system that answers questions by first building a **knowledge graph** from a collection of text documents, then walking that graph to find the passages relevant to a question. It only understands text. We add **images** to it. Each image becomes a node in the graph, and selected objects inside images (for example, the bird in a photo of a bird) are cut out and linked to the entities they show. When a question comes with a photo, the photo can then point the graph walk at the right part of the graph. The question our paper studies is **which entities are worth looking for in the images**. Our answer is to use the entity *types* (bird, building, painting, …) that MemGraphRAG's language model already assigns, instead of the coarse labels a standard named-entity tagger gives.

---

## 2. Background you need

### 2.1 Retrieval-augmented generation (RAG)

A language model only knows what was in its training data. **RAG** fixes this by first *retrieving* relevant passages from our own document collection (the **knowledge base**, KB) and putting them into the model's prompt, so the model answers from those passages.

Retrieval usually works with **embeddings**. An encoder model turns a piece of text (or an image) into a list of numbers, a vector. Things with similar meaning get vectors that are close together. To retrieve, you embed the question and return the passages whose vectors are closest. We call this **dense retrieval**.

### 2.2 Graph RAG and the "random walk"

Dense retrieval struggles with questions that need two facts joined together ("Where was the director of film X born?"). **Graph RAG** systems help with this. At indexing time a language model reads every passage and extracts facts as triples, for example `(Film X, directed by, Person Y)`. Entities (`Film X`, `Person Y`) and passages become **nodes** in a graph, with edges between entities that appear in the same fact and between entities and the passages that mention them.

To answer a question, these systems use **Personalized PageRank (PPR)**. Picture a walker moving around the graph:

- the walker starts on a few **seed** nodes, the nodes that best match the question;
- at each step it either follows a random edge or jumps back to one of the seeds;
- after many steps, nodes the walker visits often get high scores.

Passages that are connected to several seeds score high, which is how two-hop questions get answered. The main point for us is this: **anything that can become a seed can steer the ranking.** We do not need a new retrieval algorithm to add images. We only need to turn images into seeds.

Both systems we build on come from the same family (HippoRAG), so they share this machinery.

### 2.3 MemGraphRAG (the system we extend)

MemGraphRAG (KDD 2026; [code](https://github.com/XMUDeepLIT/MemGraphRAG)) builds its graph in layers:

- **facts** extracted from passages, as above;
- a **type** for each entity (a "schema"), assigned by the language model from context, for example `<Person>` or `<Animal>`;
- **conflict resolution**: when two extracted facts contradict each other, it decides which one to keep before building the graph.

At question time it embeds the question, finds facts that match it well enough (above a **fact threshold**), and uses their entities as seeds. If *no* fact matches well enough, it gives up on the graph and falls back to plain dense retrieval. This fallback matters for us (§3.2).

### 2.4 MG²-RAG (the closest existing work)

MG²-RAG (ECCV 2026; [code](https://github.com/Daboolu/MG2-RAG)) already adds images to a graph RAG system. It:

- adds image nodes to the graph;
- cuts objects out of images with **SAM3**, a model that finds and outlines objects matching a short text prompt such as "bird";
- links each cut-out (a **crop**) to an entity;
- embeds images and crops with **EVA-CLIP**, an encoder that puts images and short texts into the same vector space;
- uses image matches as extra PPR seeds.

So "images as seeds" is **not** our invention, and we say so. What differs is how each system decides which entities to look for in an image. MG²-RAG uses a standard named-entity tagger (spaCy) and keeps only entities tagged as a person, organisation, place, building, product or work of art. Animals and plants have no such tag, so they are never looked for. Its own demo (a photo of an impala) only works because the tagger mistakes "Impala" for a person's name. With correct tags it finds nothing.

### 2.5 Why this matters for our data

Our main dataset (§5) asks questions about a photo, such as a picture of a bird with the question "What is the wingspan of this bird?". The text of the question doesn't say *which* bird. A text-only system cannot know, however good its graph is. The photo is the only clue, so it has to enter retrieval somehow.

---

## 3. What we build

### 3.1 Building the index (once per collection)

1. **Text graph, unchanged.** Run MemGraphRAG exactly as released: extract facts, assign types, resolve conflicts, build the graph. We pass in one KB passage at a time (MemGraphRAG's own loader cuts across document boundaries, which loses track of which image belongs to which page).
2. **Image nodes.** Each KB image becomes a node, connected to every passage of the page it belongs to.
3. **Grounding (our contribution).** For each image, decide which of its page's entities to look for, and ask SAM3 to find them:
   - Take the entities that appear in that page's passages and exist as nodes in the final graph.
   - Keep the ones whose type is **visual**, meaning you could see one in a photo. "Bird" and "building" are visual; "date" and "law" are not. We decide this once per type, with one yes/no question to the language model per type, checked by a person, and publish the list.
   - SAM3 works best with short, generic prompts ("bird"), not names ("Eurasian blue tit"). So each visual type is mapped to a generic noun phrase, and that phrase is the prompt.
   - Each crop SAM3 returns is **linked** to the entity it was looking for. If several entities share the same phrase (two plants on one page), the crop goes to the page's main (title) entity if it is one of them. Otherwise it is shared equally between them.
   - Crops do **not** add edges or nodes to the graph. They are only used to create seeds (§3.2). This keeps the graph identical across the comparisons in §4, so differences come only from which entities were selected. MG²-RAG does add edges here; we state that difference.
4. **Image embeddings.** Embed every image and crop with a CLIP-style encoder, stored separately from the text embeddings. Text and image vectors are never compared with each other directly.

### 3.2 Answering a question

1. **Text seeds:** exactly as MemGraphRAG does it (matching facts and passages).
2. **Image seeds:** embed the question's photo.
   - The most similar KB images become seeds themselves, weighted by how similar they are.
   - The most similar crops make their **linked entities** seeds.
   - How many to take (*k*) is tuned on a development set.
3. **Mixing the two:** each set of seeds is first rescaled so its weights add up to 1, then the two are mixed with a knob **λ** (lambda):

   ```
   seeds = (1 − λ) · text seeds + λ · image seeds
   ```

   λ = 0 means text only and λ = 1 means image only. Without the rescaling, text seed weights grow with the size of the collection, and λ would mean something different for every question. λ is tuned on the development set only.
4. **Don't give up on the graph:** if the photo produced seeds, the graph walk runs even when no text fact matches the question. This is the usual case for "what is this bird?" questions, and the released MemGraphRAG would fall back to plain dense retrieval there. If the photo produced no seeds, the system behaves exactly like released MemGraphRAG.
5. **Answer:** the five top-ranked passages go to a vision-language model, which writes the answer.

### 3.3 A worked example

Question: a photo of a blue tit, and "What does this bird mainly eat?"

- Text seeds are weak: "bird" and "eat" match thousands of facts, and none of them strongly.
- The photo is close to the KB image on the "Eurasian blue tit" page, and to a crop of the bird on that page. The crop was found because the entity `Eurasian blue tit` has the visual type `<Bird>`, so SAM3 was prompted with "bird".
- The image node and the linked entity become seeds. The walk spreads from them to the blue tit's passages, including the one about its diet.
- MG²-RAG's tagger would not have tagged the blue tit, so no crop would exist for it. Plain dense image search would also find the right page, but it can't follow links onward when the answer needs a second hop.

---

## 4. What we test

We test four hypotheses. Each has **one primary comparison**, decided in advance, so we can't go looking for whichever number happens to look good. Everything else is reported as secondary.

| | Hypothesis, in plain words | Primary comparison | Secondary |
|---|---|---|---|
| **H1** | Feeding the photo into the graph walk retrieves better pages than plain image search, and better than MG²-RAG | Ours vs dense image search with the same encoder | vs MG²-RAG; vs text-only MemGraphRAG (expected to be poor; a sanity check) |
| **H2** | Using images directly works better than turning them into text captions first | Ours vs MemGraphRAG with entity-aware captions | vs generic captions; with and without a caption of the question's photo; answer accuracy |
| **H3** | Choosing what to look for by entity type finds the right objects more often than choosing by tagger label | Type-guided vs tagger-label selection, measured by **grounding precision** (§6) | How many images get a correct crop; retrieval across all four selection rules |
| **H4** | Adding images does not hurt questions that are text only | Full system vs unchanged MemGraphRAG on text-only questions: no more than 3 percentage points worse | — |

**What H3 compares.** Four selection rules, all on the same entities and all prompting SAM3 the same way (type → noun phrase). The only difference is *which* entities get selected:

| Rule | Entities SAM3 looks for |
|---|---|
| none | nothing (no crops) |
| all | every entity on the page |
| tagger label (MG²-RAG style) | entities that spaCy tags as a person, organisation, place, building, product or work of art |
| type-guided (ours) | entities whose MemGraphRAG type is visual |

**Why H3 is measured on precision, not retrieval.** In both datasets, each page has one image showing one main thing. A crop of the only thing in the picture adds little beyond the whole picture, so a retrieval difference between the selection rules is unlikely to be measurable. Whether the selected crops show the right entity *is* measurable, and that is what the claim is about. Retrieval differences are still reported.

**A risk to H3, and how we handle it.** MemGraphRAG's type prompt suggests the same coarse categories the tagger uses (person, organisation, …), so its types might be just as coarse. We check this early on 50 passages. If the types turn out coarse, the type-guided rule uses a second, finer-grained typing prompt that already exists in MemGraphRAG's code. It costs one extra language-model call per passage.

---

## 5. Data

### 5.1 E-VQA (main dataset)

Questions about a photo, answered from Wikipedia. We use the version packaged in **M2KR**.

- **Questions:** 500 from the test split. The photos come from iNaturalist (animals and plants) and Google Landmarks (buildings and places), sampled in the same proportion as the full test set. A separate **development set** of 50 questions from the validation split is used for all tuning.
- **Knowledge base:** the Wikipedia pages that answer the 500 questions, plus randomly chosen other pages as distractors, up to 2,000 pages in total. Each page is split into a few passages. A "page" is identified by the passage ID with its trailing number removed.
- **KB images:** M2KR's KB is text only, so we take one image per page from MuKA's published lists of Wikimedia image links. We drop thumbnails from a web search engine and placeholder images, and record how many were dropped. Downloads respect Wikimedia's rate limits.
- **Question photos:** from `BByrneLab/M2KR_Images` on Hugging Face.
- **No near-duplicates:** if a question's photo is nearly identical to a KB image (checked with perceptual hashing), the question is removed from the main results, because otherwise the test would be "find the same picture". We report how often this happened.
- **Honest description:** questions are single-hop and mostly about recognising what is in the photo; one image per page.

### 5.2 MMQA (second dataset)

Text questions over Wikipedia pages that include images, for example "Which of these two buildings has a dome?". There is no question photo here, so the image search uses the question text.

- **Image questions:** about 440 development-set questions that need an image but not a table.
- **Text-only questions:** 300, used for H4.
- **KB:** the pages belonging to those questions, up to 2,000. One image per entity.
- **An early check decides its role:** on 100 development questions, does searching images by question text beat plain text retrieval? If yes, MMQA is used for H1–H4. If no, it is used only for H3 (precision) and H4.
- **Licence:** none is stated by the authors. We use it for research only, don't redistribute it, and say so in the paper.

### 5.3 Text check

200 HotpotQA questions from MemGraphRAG's own data. With no images, our code must give **exactly** the same results as released MemGraphRAG. This is a test that our changes didn't break anything, not a hypothesis.

### 5.4 Not used

| Dataset | Why not |
|---|---|
| InfoSeek | Adds nothing structurally new and needs a 9 GB download; added only if time remains |
| E-VQA two-hop questions | Only in the official release, with a separate 4.9 GB KB; not in M2KR |
| WebQA | About 51 GB of images; test answers only through an online leaderboard |
| CrossModalQA, RETINA | Not publicly available |

---

## 6. How we measure

- **Retrieval (the main measure):**
  - **page Recall@5**: the share of questions for which the correct page is among the top 5 results;
  - also Recall@1 and @10, and MRR (how high the correct page ranks on average).
- **Answers:**
  - **containment**: does the model's answer contain the correct answer?
  - an **LLM judge**: a different language model decides whether the answer is correct. It must differ from the answering model, because models grade their own answers leniently;
  - BEM, a learned answer-matching score used for E-VQA, if it installs on our cluster.
- **Grounding (H3):**
  - **precision**: 100 random crops per selection rule per dataset, judged by two people who don't know which rule produced them: "does this crop show the entity it is linked to?";
  - **coverage**: the share of KB images whose main entity gets at least one correct crop.
- **Is a difference real?**
  - We use paired statistical tests: a bootstrap for Recall@5 and McNemar's test for answer accuracy.
  - Because we run four primary comparisons, we tighten the threshold for "significant" (Holm correction).
  - With about 500 questions we can reliably detect differences of roughly 5 percentage points. Smaller differences are reported as "not detectable", not as "no effect".
- **Equal input for answering.** Every system gets the same kind of evidence: once text passages only, once text plus the images of the retrieved pages. A system can't look better just because it gave the answering model more to read.
- **Leakage controls.** We remove near-duplicate photos (above), never use Wikipedia's own image captions, and judge H2 on retrieval rather than answers.

---

## 7. Systems we run

| System | Why it's here |
|---|---|
| **Ours** (MemGraphRAG-V, type-guided) | Our method |
| MemGraphRAG + **generic captions** of all images, indexed as text | H2: the obvious alternative to using images directly |
| MemGraphRAG + **entity-aware captions** (the captioner also sees the page title) | H2: the strong version of that alternative, so a win means something |
| **Text-only** MemGraphRAG, as released | Sanity check and H4 |
| **Dense image search** (no graph) | H1: the simplest way to use the photo |
| **MG²-RAG**, as released, with the same encoder | H1: the closest existing system |
| **No retrieval**: the answering model alone | The floor |
| The four **selection rules** (none / all / tagger label / type-guided) | H3 |
| Ours with **λ = 0** (image nodes in the graph, but no image seeds) | Shows whether gains come from the seeds or just from the extra nodes |
| Ours **without MemGraphRAG's memory layer** (its older, simpler indexing path) | Shows whether the memory layer matters |

All captions are indexed as extra passages linked to their page. We re-run every system ourselves on our data. Our numbers are never placed next to numbers published in other papers, because the data and models differ.

---

## 8. Models and compute

Every system uses the **same model in each role**. When two systems differ in results, the cause is then the method, not the models.

| Role | Model |
|---|---|
| Building the text graph (all MemGraphRAG versions) | Qwen2.5-7B, run on our cluster |
| Text embeddings | `bge-large-en-v1.5`, as in MemGraphRAG's released code |
| Image embeddings | SigLIP2 while developing; **EVA-CLIP-8B for the final results**, for every system |
| Finding objects in images | SAM3, run on our cluster |
| Captions and answers | Qwen3-VL-8B |
| LLM judge | Qwen2.5-7B (different from the answering model) |
| Named-entity tagger (tagger-label rule) | spaCy `en_core_web_trf` |

**Why two image encoders.** SigLIP2 is small and fast, so we develop and tune with it. EVA-CLIP-8B is the encoder MG²-RAG's paper uses, so the final runs use it for every system, MG²-RAG included. That way nobody can say we weakened MG²-RAG. λ, *k* and the fact threshold are re-checked on the development set with EVA-CLIP-8B before the final runs.

**Differences from the original papers.** We are not reproducing either paper. We compare systems under identical conditions. The paper will include this table:

| | MemGraphRAG paper | MG²-RAG paper | Ours (every system) | Why |
|---|---|---|---|---|
| Building the graph | gpt-4o-mini | spaCy tagger (no LLM) | Qwen2.5-7B | Runs on our cluster at no cost |
| Text embeddings | NV-Embed-v2 | EVA-CLIP-8B | bge-large | MemGraphRAG's released default; CLIP encoders only read 64–77 tokens of text |
| Image embeddings | — | EVA-CLIP-8B | EVA-CLIP-8B | Same as MG²-RAG |
| Answering | gpt-4o-mini | Qwen2.5-VL-7B | Qwen3-VL-8B | Must read images; newer and openly licensed |
| Data size | 1,000 questions per dataset | Full E-VQA test, 100k-page KB | 500 questions, ≤ 2,000-page KB | Compute and time |

**Known risk from this.** A 7B model extracts fewer facts than gpt-4o-mini. With fewer facts, MemGraphRAG falls back to dense retrieval more often, which could make image seeds look more or less helpful than they would with a stronger model. As a robustness check, we rebuild the KB for 100 E-VQA questions with gpt-4o-mini (≈ $2–5) and check that the H1 comparison points the same way.

**Compute.** Everything runs on the Snellius cluster:
- development and tuning on the free half-A100 slice;
- building the main indexes and the final EVA-CLIP-8B runs on full A100/H100 GPUs, paid from the shared course credit account, with a budget of 5,000 credits (about 39 A100-hours);
- logging the credits for every paid job.

Indexing is the slowest step. We speed it up with a quantised Qwen2.5-7B and by reusing the fixed part of the prompt across calls.

---

## 9. Plan

### 9.1 Early checks, each with a fallback

| By | Check | If it fails |
|---|---|---|
| ✅ done | MemGraphRAG runs end to end on Snellius with Qwen2.5-7B | — |
| ✅ done | SAM3 access granted | — |
| 7 Oct | Are MemGraphRAG's types finer than tagger labels? (50 E-VQA passages) | The type-guided rule uses the finer-grained typing prompt |
| 7 Oct | MMQA: does searching images by question text beat text retrieval? | MMQA used only for H3 precision and H4 |
| 7 Oct | Qwen3-VL-8B fits on the free GPU slice; EVA-CLIP-8B and SAM3 fit together on one full A100 | Answer on a paid full A100; use SigLIP2 everywhere |
| 11 Oct | The full image layer runs end to end on the E-VQA development set | Switch to the minimal paper (§9.3) |

### 9.2 Timeline

| Dates | Work |
|---|---|
| 4–7 Oct | Early checks; type → noun-phrase list; SAM3 trial on a few E-VQA images; measure indexing speed |
| 5–8 Oct | Data preparation (E-VQA with KB images and duplicate removal; MMQA subset); loader; metric and statistics scripts |
| 5–11 Oct | Image layer: image nodes, selection rules, crop seeds, seed mixing, no-give-up rule |
| 8–12 Oct | Build indexes, including caption versions; dense image search and no-retrieval baselines; MG²-RAG |
| 9–12 Oct | Two people judge crop precision |
| 12–16 Oct | EVA-CLIP-8B embeddings; re-tune on development set; main runs and ablations. **Results frozen on 16 Oct.** |
| 5–22 Oct | Writing (background and method start first) |
| 23 Oct | Submission |

### 9.3 If we run out of time

- **Minimal paper:** E-VQA only, with H1 against dense image search, H2 against both caption baselines, and H3 as crop precision and coverage.
- **Drop in this order:** λ sweep → EVA-CLIP-8B final runs (report SigLIP2 instead) → InfoSeek → MG²-RAG on MMQA → "without memory layer" run → finer-grained types (unless the early check required them) → MG²-RAG on E-VQA.
- **Never drop:** the caption baselines, crop precision for H3, and H4.

---

## 10. Limitations we will state

- E-VQA questions are single-hop and mostly about recognition, and each page has one image. This limits how much the graph can help, and how much crop selection can affect retrieval.
- MMQA has no question photos, so its image search depends on matching question text to images, which CLIP-style encoders do poorly.
- We use a 7B model to build the graph, not gpt-4o-mini, and smaller collections than the papers. Our absolute numbers are not comparable with theirs.
- MemGraphRAG's released code does not implement two retrieval formulas from its paper (type-node seeding and an IDF weight). We use the released code as it is and call it "MemGraphRAG (released code)".
- Using images as PPR seeds, and comparing images with captions, have been explored before (MG²-RAG; HVM-GraphRAG, RAG-Anything and mKG-RAG for captions). Our contribution is the **selection of which entities to ground**, using the language model's entity types, and a controlled comparison of the alternatives.

---

## Glossary

| Term | Meaning |
|---|---|
| Knowledge base (KB) | The document collection the system retrieves from |
| Passage / page | A passage is a short piece of a Wikipedia page; a page is the whole article |
| Embedding, encoder | A vector representing text or an image; the model that produces it |
| Dense retrieval | Returning the items whose embeddings are closest to the query's |
| Triple, fact | `(subject, relation, object)` extracted from text |
| Entity type / schema | The category MemGraphRAG's LLM assigns to an entity, e.g. `<Bird>` |
| Named-entity tagger | A standard tool (spaCy) that labels names as person, organisation, place, … |
| PPR, seed | The graph walk used for ranking; seeds are the nodes the walk starts from and returns to |
| Fact threshold | How well a fact must match the question before MemGraphRAG uses it as a seed |
| SAM3 | A model that finds and outlines objects in an image given a short text prompt |
| Crop, grounding | A cut-out of an object in an image; grounding means linking it to an entity |
| CLIP-style encoder | Embeds images and short texts into the same space (SigLIP2, EVA-CLIP-8B) |
| λ (lambda), *k* | The text–image mixing knob; how many top image/crop matches become seeds |
| Recall@5 | Share of questions with the correct page in the top 5 |
| Development set | A small separate set of questions used only for tuning, never for reported results |
| Ablation | A run with one part removed, to see what that part contributes |
