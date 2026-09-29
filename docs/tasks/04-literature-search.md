# Task 4: Find missed information and published work

**Goal:** make sure no published paper already does "memory/graph RAG + native image evidence + ontology-guided grounding", and collect what we must cite and compare against.

**Done when:** `docs/related-work.md` lists every relevant paper with a one-line relevance note and a verdict (cite / compare against / threatens novelty), and any novelty threat is raised with the group.

> The seed list below was written from memory during the audit and **has not been checked**. Verify every title, venue and year against the actual paper before citing. Add arXiv IDs as you verify them.

## Seed list (unverified)

| Area | Works to look up |
|---|---|
| Graph / memory RAG (text) | HippoRAG (NeurIPS 2024); HippoRAG 2 ("From RAG to Memory"); Microsoft GraphRAG ("From Local to Global"); LightRAG; RAPTOR; G-Retriever |
| Multimodal graph RAG | MMGraphRAG and other "multimodal knowledge graph RAG" papers; anything citing MG²-RAG |
| Knowledge-based VQA with retrieval | Encyclopedic-VQA (E-VQA); InfoSeek; OVEN; FLMR / PreFLMR and the M2KR benchmark; EchoSight; Wiki-LLaVA; ReflectiVA |
| Multimodal RAG generally | MuRAG; RA-CM3; VisRAG; ColPali; MRAG-Bench; surveys of multimodal RAG |
| Grounding / encoders | SAM, SAM 2, SAM 3; GroundingDINO; OWL-ViT / OWLv2; CLIP; EVA-CLIP; SigLIP 2 |
| Conflict / memory in RAG | Knowledge-conflict surveys; papers on contradiction detection in retrieved context |

## Search strategy

1. **Forward citations** of both base papers (Google Scholar / Semantic Scholar "cited by"). They are recent, so also search arXiv listings from the last months directly.
2. **Keyword queries** (arXiv, ACL Anthology, Semantic Scholar): `multimodal graph RAG`, `knowledge graph retrieval images PageRank`, `visual entity grounding retrieval augmented generation`, `ontology guided grounding`, `schema-aware multimodal retrieval`, `memory graph multimodal RAG`.
3. **Benchmarks:** papers reporting on InfoSeek / E-VQA / M2KR with graph-based retrieval — these are our most direct competitors and possible extra baselines.
4. **Check the audit's [I] claims** that depend on outside facts: SAM3 licence terms, whether InfoSeek/E-VQA images are available without the full OVEN/AToMiC downloads, EVA-CLIP-8B VRAM use, current gpt-4o-mini pricing.

## Output format for `docs/related-work.md`

| Paper | Venue, year | Link | What it does (1 line) | Verdict |
|---|---|---|---|---|

Verdicts: **cite** (background), **compare** (should be a baseline or discussed in results), **threat** (overlaps our novelty — tell the group immediately).
