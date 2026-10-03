# Task 1: Snellius smoke test

**Goal:** a real (non-stubbed) MemGraphRAG index + QA run on a tiny HotpotQA slice, on Snellius. This is the week-1 gate (target **2 Oct**). Stretch goal: the same for MG²-RAG's impala demo.

**Done when:** MemGraphRAG answers ~10 HotpotQA questions end-to-end and you've written down in [the log section below](#log) every command that deviated from this guide.

> This guide is **untested on Snellius.** It is derived from the audit's local stub runs ([audit §3](../audit/audit-report.md#3-getting-each-codebase-running)). Fix it as you go and commit the fixes.

## Before you start

- Read the course's *How to use Snellius* guide (on Canvas). Use the **course reservation** on `gpu_mig` (free, ½ A100) for everything here; the reservation name is in that guide. Don't put it in this public repo.
- Partition ladder from the guide: `int3` (free, ⅐ A100, CLI only) → `gpu_mig` + reservation (free) → `gpu_mig` (64 credits/h) → `gpu_a100` (128/h) → `gpu_h100` (192/h). Check budget with `accinfo`.
- The guide advises **against Anaconda** on Snellius. Use `module load` + `python -m venv`.
- Decide with the group which LLM to use: an OpenAI key (gpt-4o-mini; this smoke test is < US$1) or a self-hosted model via vLLM. Never commit keys.

## What will be downloaded

| Item | Size | Where |
|---|---|---|
| `BAAI/bge-large-en-v1.5` (safetensors only) | ≈1.3 GB | scratch |
| MemGraphRAG repo + bundled data | small | home |
| (stretch) `spacy/en_core_web_trf` | 0.46 GB | scratch |
| (stretch) `facebook/sam3` `sam3.pt` | 3.45 GB, **gated, manual approval** | scratch |
| (stretch) `BAAI/EVA-CLIP-8B` | 32.9 GB | scratch — check `/projects/2/managed_datasets` first |

## Steps: MemGraphRAG

```bash
ssh <user>@snellius.surf.nl
module purge
module load 2023            # then `module avail Python` and pick a 3.10/3.11 Python module
python -m venv ~/venvs/mgr && source ~/venvs/mgr/bin/activate

git clone https://github.com/XMUDeepLIT/MemGraphRAG.git && cd MemGraphRAG
# requirements.txt pins vllm (not needed for online OpenIE) — install everything else:
grep -v '^vllm' requirements.txt > req-novllm.txt && pip install -r req-novllm.txt
# gritlm is imported eagerly by src/embedding_model/__init__.py:4 even when using BGE;
# it is in requirements.txt, so it should be installed now. If it fails, comment out that import.

huggingface-cli download BAAI/bge-large-en-v1.5 --include "*.json" "*.safetensors" "*.txt" \
  --local-dir /scratch-shared/$USER/models/bge-large-en-v1.5   # or any scratch dir you own
```

Build a tiny corpus (first 10 HotpotQA questions and their context passages). The corpus is a plain `.txt`; `code/index.py` re-chunks it into 256-token windows.

> For the smoke test this is fine. For evaluation, do **not** go through `index.py`: it slides windows across document boundaries and drops document IDs. Index one passage per chunk from `dataset/*/*_corpus.json` instead (methodology audit A13).

```bash
mkdir -p smoke
python - <<'EOF'
import json
qs = json.load(open("dataset/hotpotqa/hotpotqa.json"))[:10]
json.dump(qs, open("smoke/questions.json", "w"), indent=1)
seen, out = set(), []
for q in qs:
    for title, sents in q["context"]:
        if title not in seen:
            seen.add(title); out.append(title + "\n" + "".join(sents))
open("smoke/corpus.txt", "w").write("\n\n".join(out))
print(len(qs), "questions,", len(out), "passages")
EOF
```

Run indexing, then QA, **directly with python** — `code/run_index.sh` and `code/run_retrieval_test.sh` crash on an undefined `$PYTHON` and contain the authors' hard-coded paths and a third-party proxy URL.

```bash
export OPENAI_API_KEY=...        # or set --llm-base-url to your vLLM server
EMB=/scratch-shared/$USER/models/bge-large-en-v1.5

python code/index.py --corpus smoke/corpus.txt --save-dir outputs/smoke \
  --llm-name gpt-4o-mini --embedding-model $EMB --tokenizer $EMB

python code/retrieval_dataset_test.py --questions smoke/questions.json \
  --save-dir outputs/smoke --output results/smoke/qa.json \
  --llm-name gpt-4o-mini --embedding-model $EMB --fact-similarity-threshold 0.4
```

Run both inside a job (`sbatch` with `--partition=gpu_mig --gpus=1 --time=00:30:00` and the reservation), or interactively via `salloc`. The embedder is the only GPU user here.

## What to record

- [ ] Exact module names, Python and torch versions that worked
- [ ] Wall-clock time and number of LLM calls/tokens for indexing (the code logs token counts per stage)
- [ ] Counts from `outputs/smoke/`: facts before/after conflict resolution, schemas kept, node/edge counts
- [ ] **Schema-type histogram (decides H3; see [methodology audit A5](../audit/methodology-audit-2026-10-03.md#2-findings-and-proposed-changes)).** MemGraphRAG's schema prompt suggests the 18 OntoNotes NER labels. Record whether the LLM also produces finer types (animal, species, building…):
  ```bash
  python - <<'PY'
  import json, collections
  m = json.load(open("outputs/smoke/initial_memory_with_schema.json"))
  c = collections.Counter()
  for s in m["schema_layer"]:
      h, _, t = s["content"]; c[h] += s.get("frequency", 1); c[t] += s.get("frequency", 1)
  print(len(c), "distinct types"); print(c.most_common(40))
  PY
  ```
  HotpotQA has few visual entities, so repeat this on ~50 M2KR E-VQA passages as soon as the data builder exists.
- [ ] Whether the answers in `results/smoke/qa.json` look right (the runner computes **no** metrics; eyeball containment of the gold `answer`)
- [ ] Anything that contradicts the audit — open an issue

## Stretch: MG²-RAG impala demo

Needs CUDA + CuPy, spaCy on GPU, EVA-CLIP-8B and SAM3. By default it expects **2 GPUs**; on one set `MG2RAG_SAM3_DEVICE=0`. With EVA-CLIP-8B resident, a single-GPU run likely needs a full A100 (40 GB), i.e. `gpu_a100`, not the MIG slice [I]. `ijson` is missing from its `requirements.txt`. Follow [its README](https://github.com/Daboolu/MG2-RAG) and note that `examples/showcase` was *not* produced from the bundled input. Only attempt this after SAM3 access is approved and the group agrees to spend credits.

## Log

Add dated notes here (who, what worked, what didn't).
