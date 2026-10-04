# Data licences and release status: MMQA, MuKA/Wikimedia, CrossModalQA, RETINA, WebQA (4 Oct 2026)

Builds on `docs/audit/research-findings-2026-10-03.md` §3.3 and §6. Tags: **[V-read]** I read it in the primary source at the cited URL. **[V-run]** I computed it from downloaded files or API calls. **[I]** inferred. I did not modify the repo. Small downloads went to the scratchpad: MMQA dev and image metadata (4 MB), the MMQA and MuKA PDFs, MuKA URL lists (2 MB), 8 MMQA images (16 MB, read from the remote zip with HTTP range requests).

Not legal advice. The legal points below are about research use only. We do not redistribute anything.

---

## 1. MultimodalQA (MMQA): licence

| Item | Finding | Tag |
|---|---|---|
| GitHub licence field | `gh api repos/allenai/multimodalqa` returns `"license": null`. The repo has no LICENSE file: its root holds only `.gitignore, README.md, _config.yml, baselines, dataset, deps, figures, index.md`. Last push was 2022-10-12. | [V-run] |
| README / project page | The README ([github.com/allenai/multimodalqa](https://github.com/allenai/multimodalqa)), `index.md` and [allenai.github.io/multimodalqa](https://allenai.github.io/multimodalqa/) say nothing about a licence, terms of use or copyright. | [V-read] |
| Licence PR | [PR #4](https://github.com/allenai/multimodalqa/pull/4) "Added Apache License pursuant to Allen AI Open Source Policy" came from an outside user (Bhaney44) on 2021-05-25. It is still **open and unmerged**, so no licence was ever adopted. | [V-run] |
| AllenAI dataset page | `https://allenai.org/data/multimodalqa` returns **404**. MMQA appears neither on allenai.org/data nor in the site's sitemap. | [V-run] |
| Paper | ICLR 2021, [arXiv 2104.06039](https://arxiv.org/abs/2104.06039). The PDF only says "Our dataset and code are available at https://allenai.github.io/multimodalqa". It contains no licence or terms statement (I grepped the full text for licen/copyright/terms/redistribut). | [V-run] |
| HF mirrors | Third-party copies label themselves differently. [`BiXie/multimodalqa`](https://huggingface.co/datasets/BiXie/multimodalqa) and [`JoohyungYun/multimodalqa_doc`](https://huggingface.co/datasets/JoohyungYun/multimodalqa_doc) say `apache-2.0`. [`TableQAKit/MMQA`](https://huggingface.co/datasets/TableQAKit/MMQA) states no licence. None of them is from AllenAI, so their labels **carry no authority** over the questions or the images. | [V-run] |
| Images (`final_dataset_images.zip`) | 2,362,869,144 bytes, last modified 2021-04-21, 57,063 zip entries (S3). The paper's appendix says each image comes from the Wikipedia page of its WikiEntity via the MediaWiki API ("profile images", §2.2 and App. A). Every image keeps the licence of its original Wikipedia or Commons file. Many MMQA image questions ask about **posters (182 of the 940 dev image questions) and logos (179)**. On English Wikipedia these are usually locally hosted **non-free (fair-use) files** ([WP:NFC](https://en.wikipedia.org/wiki/Wikipedia:Non-free_content)), not Commons CC files. | [V-read] paper / [V-run] counts / [I] non-free share |

**Can a student group use it for a research paper?** [I] Yes, for internal experiments.
- The annotations were published openly for research, with a public leaderboard. Using them only means downloading them, which is the use the authors intended. With no licence there is no grant of redistribution rights, but we do not redistribute.
- For the images, the EU DSM Directive 2019/790 **Art. 3** requires member states to allow "reproductions and extractions made by research organisations … for the purposes of scientific research, text and data mining of works … to which they have lawful access". Copies must be "stored with an appropriate level of security" ([EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019L0790)) [V-read]. TU/e is a research organisation. Whether coursework by students falls under "research organisation" is not settled [I].
- Practical rules [I]:
  1. Do not re-publish the MMQA image zip or the annotations.
  2. In the paper, cite MMQA and say "licence not specified by the authors".
  3. Do not reproduce non-free posters or logos in paper figures. Pick Commons images for figure examples and attribute them per file.

**Dev counts (re-confirmed)** [V-run on `dataset/MMQA_dev.jsonl.gz`]: 2,441 dev questions. **940** have `image` in `metadata.modalities`. 863 of those 940 have exactly one gold image doc; 77 have 2 to 8. Per-question context offers 2 to 15 image docs (mean 10.5). 429 of the 940 involve `ImageListQ`.

| Type | Count |
|---|---|
| ImageQ | 230 |
| Compose(TableQ,ImageListQ) | 195 |
| Compose(ImageQ,TableQ) | 142 |
| ImageListQ | 141 |
| Compare(Compose(TableQ,ImageQ),TableQ) | 104 |
| Compose(TextQ,ImageListQ) | 46 |
| Intersect(ImageListQ,TableQ) | 44 |
| Compose(ImageQ,TextQ) | 20 |
| Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ)) | 15 |
| Intersect(ImageListQ,TextQ) | 3 |

The test split has no answers. [Issue #10](https://github.com/allenai/multimodalqa/issues/10) asks how to get them and has no maintainer reply [V-run].

## 2. MMQA: images per entity, and what the images show

**One image per entity** [V-run on `MMQA_images.jsonl.gz`, fields `title, url, id, path`]:
- 57,058 images, 56,571 distinct titles, 56,162 distinct Wikipedia URLs.
- **56,200 titles (99.3%) have exactly 1 image**, 318 have 2, and 53 have 3 to 7. The few duplicates come from in-table images, for example NASCAR drivers with 6 to 7 images.
- For the gold images of the 940 dev image questions, the number of corpus images with the same title is **1 in 1,118 cases**, 2 in 9 cases and 5 in 1 case.

The paper says the same thing: "To associate entities with their representative image, we map entities and their profile images in their Wikipedia pages. Overall, we obtain 57,713 images, with 889 in-table images and 56,824 WikiEntities images" (§2.2). App. A says each image "is associated with a specific WikiEntity". Crowd workers saw "an image alongside its WikiEntity" and were asked for a question "with the entity being the focus" (§2.3) [V-read, [arXiv 2104.06039](https://arxiv.org/abs/2104.06039)].

**What the images show** [V-run, small sample]: I range-read 8 gold images from the remote zip and viewed 6:
- Single subject: Richard Strauss portrait (for "Arabella"), Miguel Cabrera at bat, the "Girl Meets World" logo.
- Painting with 2 figures: Don Giovanni.
- Multi-object scenes: Parkersburg cityscape (river, bridges, buildings) and the Nagyatád chapel (tower, crucifix, statue, fence).

Question wording over the 940 dev image questions (keyword counts): poster 182, logo 179, player 79, wearing 70, cover 65, hair 59, building 19, statue 13. The questions target **attributes or parts of the single depicted subject**, for example "What is Don Giovanni holding in his right hand?", "…wearing what on his hands?", "a white tower with an arched entrance, and a cross-shaped statue out in front".

**Implication for SAM3 sub-entity grounding** [I]:
- Each image maps 1:1 to one entity document, so image retrieval is close to entity retrieval. A crop rarely names a *different* KB entity: gloves, a sword or a fountain are not MMQA documents.
- Crops might help with attribute questions (localising "right hand" or "logo element") and with the minority of city or building scenes.
- They are unlikely to add **new graph links** between documents, so H3-style gains from cross-entity grounding are structurally limited on MMQA.
- About 40% of image questions involve posters, logos or covers, which are graphic designs. SAM3 concept prompts on these are of doubtful value.

**A re-crawled multi-image variant exists** [V-run partial]: [`JoohyungYun/multimodalqa_doc`](https://huggingface.co/datasets/JoohyungYun/multimodalqa_doc), created 2026-02-09, labelled apache-2.0 by the uploader.
- It is a full-page re-crawl of 3,236 MMQA-related Wikipedia documents with in-article images, captions and section `heading_path`: 9,805 image components and a 5.48 GB `image_dump.parquet`.
- In the 6,400 of 9,805 rows I fetched through datasets-server (stopped by rate limiting), 1,037 docs have images. The median is 3 images per doc, the mean 6.2.
- Only 86 rows carry a `label_id`, and **all 86 match MMQA image IDs**. So the original MMQA profile images are mostly *not* linked in this variant.
- Provenance and associated paper are unknown. It is too large for a casual download, but the gold images could be fetched with parquet range reads [I].

## 3. MuKA image-URL lists and Wikimedia practice

**MuKA licence** [V-read/V-run]:
- [github.com/lhdeng-gh/MuKA](https://github.com/lhdeng-gh/MuKA): GitHub API `license: null`, no LICENSE file, last push 2025-04-06.
- README disclaimer: "The images we have collected are for research purposes only, and we shall not be held responsible for any issues arising from their use."
- The COLING 2025 paper ([aclanthology.org/2025.coling-main.647](https://aclanthology.org/2025.coling-main.647/)) has no licence or terms statement (full-text grep).
- So the URL lists themselves are unlicensed, but they are lists of URLs plus entity names. Each image's licence is that of its Wikimedia file [I].

**URL composition** [V-run on `data_preparation/*_passages_image_urls.jsonl.gz`]:

| List | Total | Wikimedia | Bing thumbnails | Null | Commons paths | `/wikipedia/en/` local files | Other |
|---|---|---|---|---|---|---|---|
| E-VQA | 19,408 | 19,388 | 20 (`tse*.mm.bing.net`) | 0 | 18,886 | 127 | 395 |
| InfoSeek | 34,332 | 32,135 | 2,192 | 5 | 23,500 | 4,286 | 6,541 |

- Local `/wikipedia/en/` files are typically non-free [I]. In E-VQA, "other" is mostly `commons.wikimedia.org/wiki/Special:FilePath` redirects.
- The Bing thumbnails are scraped web-search results with unknown rights and provenance. Exclude or flag them [I].

**Wikimedia rules for downloading** [V-read]:
- **User-Agent**: the [Wikimedia Foundation User-Agent Policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy) (meta page redirects there) requires an informative User-Agent with contact information. Example: `CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0`.
  - "Do not use generic agents such as 'curl', 'lwp', 'Python-urllib'". Non-descriptive defaults like `python-requests/x` "may also be blocked".
  - "Do not copy a browser's user agent for your bot".
  - Empty or generic agents get HTTP 403. Including "bot" in the string is encouraged.
- **Rate limits**: the [Wikimedia Robot policy](https://wikitech.wikimedia.org/wiki/Robot_policy), "Media API rules", covers `https://upload.wikimedia.org/…`:
  - "Always keep a total concurrency of at most 2, and limit your total download speed to 25 Mbps (as measured over 10 second intervals)."
  - "Only use originals or one of our standard thumbnail sizes" and "Prefer thumbnails to downloading of originals if possible."
  - General rules: honour `429` plus `Retry-After`, honour robots.txt, and keep overall concurrency below 10 and below 20 req/s.
  - Limits are "global for all Wikimedia properties". Toolforge/WMCS bots are exempt.
- **Per-image licences**: [Commons:Reusing content outside Wikimedia](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) says almost all Commons content "may be freely reused subject to certain restrictions", and "each may have different requirements for crediting". English-Wikipedia local files are governed by [WP:Non-free content](https://en.wikipedia.org/wiki/Wikipedia:Non-free_content) (fair use).
- **Practice for us** [I]:
  - Download standard-size thumbnails, for example 512 px wide (MuKA rescaled to a 512 px short side anyway).
  - Use concurrency of at most 2, a descriptive UA with a contact email, and back off on 429.
  - Keep files private (DSM Art. 3(2) "appropriate level of security").
  - Attribute any image shown in the paper per its file page.

## 4. CrossModalQA and RETINA: release status

**CrossModalQA** ([arXiv 2609.05518](https://arxiv.org/abs/2609.05518), Cai, Hong, Yuan, Zhou, Zhang, Huang):
- **Still only v1 (2026-08-31)** [V-run arXiv API]. The arXiv licence is "arXiv.org perpetual non-exclusive license" [V-read]. That covers the paper only and grants no data rights.
- **No data or code link** in the v1 HTML: no github, huggingface or project URLs in the body. The only external links are a NeurIPS DOI in the references, an OpenAI system card and fonts [V-run].
- No GitHub repo found: `gh search repos CrossModalQA` returns 0. Code search for "2609.05518" finds only arXiv-digest repos [V-run].
- No HF dataset found: dataset search "crossmodal" lists none. `huggingface.co/api/arxiv/2609.05518/repos` returns empty, and HF Papers says "Paper not found" [V-run].
- Papers with Code is gone: paperswithcode.com 302-redirects to huggingface.co/papers/trending [V-run].
- Gold evidence exists by design: each question retains "explicit supporting evidence" as a "supporting subgraph" [V-read v1 HTML]. Size is 1,863 QAs, 4,987 articles and 4,431 Commons images (abstract).
- **Verdict: not released.**

**RETINA** ([arXiv 2511.22843](https://arxiv.org/abs/2511.22843), Lee et al., Korea Univ./KAIST):
- Latest is **v2 (2026-02-25)**. v1 links the project page [leeds1219.github.io/RETINA](https://leeds1219.github.io/RETINA/) [V-run].
- **Code repo exists**: [github.com/leeds1219/RETINA](https://github.com/leeds1219/RETINA), **MIT** licence (GitHub API). It holds only `LICENSE, README.md, assets/` (8 MB of figures). Last commit 2026-02-26 ("Add TODO list") [V-run].
- README states: "TODO-List – [ ] Release RETINA bench." It links "[Access the RETINA Dataset](https://huggingface.co/datasets/Lee1219/RETINA)", but that URL's API returns **401**, meaning it is private or missing. User `Lee1219` has **0 public datasets**; their 3 models are unrelated retrievers [V-run].
- README terms: "intended for **non-commercial research purposes**". Document images come from MuKA URLs plus an `images.zip` on that (unavailable) HF repo [V-read].
- Gold evidence: each query has a target Wikipedia document and is evaluated by Recall@k, so gold doc IDs exist by design [V-read v2 HTML]. Size is 120k train and 2k human-curated test (abstract).
- **Verdict: announced, not released as of 4 Oct 2026.**

## 5. WebQA test labels (lower priority)

- [WebQnA/WebQA](https://github.com/WebQnA/WebQA) README (CC0-1.0) ships test as a separate `WebQA_test.json` (7,540 samples). It defines a submission "Output Format" (`{guid: {sources, answer}}`) [V-read].
- The project page [webqna.github.io](https://webqna.github.io/) links an [EvalAI challenge 1255](https://eval.ai/web/challenges/challenge-page/1255/overview). The EvalAI API shows phase "WebQA" active, public, ending 2099-02-24 [V-run].
- **Test labels are hidden. Scoring is via EvalAI submission only** [I, strongly supported]. I did not download the test JSON from Google Drive to confirm the absence of `A`/`sources` fields.
- Use train/val (with `sources` gold IDs) for local evaluation.

---

## Summary of verdicts

- **MMQA: usable** for internal research. There is no licence: repo `null`, the Apache PR is unmerged, the AllenAI page is 404. Data is public, intended for research, and we only download. Images keep their per-file Wikipedia licences, with many non-free posters and logos, so do not redistribute them or show them in figures. Dev has 940 image questions (863 with 1 gold image).
- **MMQA as a test of SAM3 sub-entity grounding: weak.** 99.3% of entities have exactly 1 "profile image", and 1,118 of 1,128 dev gold images are the sole image of their entity. Questions ask about attributes of that one subject, so crops rarely link to other KB entities. A multi-image full-page re-crawl (`JoohyungYun/multimodalqa_doc`, 5.5 GB, provenance unclear) exists.
- **MuKA URL lists: usable.** There is no licence, only a "research purposes only" disclaimer. They are URLs, and images follow per-file Wikimedia licences. Drop the 20 E-VQA and 2,192 InfoSeek Bing thumbnails. Download with a descriptive User-Agent, at most 2 concurrent requests, at most 25 Mbps, standard thumbnail sizes, and honour 429.
- **CrossModalQA: not usable (not released).** Only v1 exists, with no code or data link in the paper, GitHub, HF or HF Papers.
- **RETINA: not usable yet.** The MIT repo has only a README. The "Release RETINA bench" TODO is open and the HF dataset `Lee1219/RETINA` returns 401. Terms will be non-commercial research.
- **WebQA: usable for train/val** (CC0, gold `sources`). Test labels are hidden behind EvalAI (open until 2099). The image archive is about 51 GB.
