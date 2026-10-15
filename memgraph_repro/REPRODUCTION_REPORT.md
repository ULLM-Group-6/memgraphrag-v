# MemGraphRAG reproduction on Snellius: procedure and results

Date: 12 October 2026
Upstream: https://github.com/XMUDeepLIT/MemGraphRAG, commit `cd6fabda1ec31a302bb283089e17c087ab7713e8`
Scope: the repository's own README pipeline, run on the four datasets bundled in its `dataset/` folder (medical, 2WikiMultiHopQA, HotpotQA, MuSiQue). Nothing else was added to the data.

---

## 1. Summary

MemGraphRAG runs end to end on Snellius for all four bundled datasets with a free, self-hosted LLM (Qwen2.5-7B-Instruct served by vLLM) instead of the OpenAI model used in the paper. Numbers are lower than the paper's, as expected with a much smaller model.

String accuracy (gold answer contained in the predicted answer, word-boundary match, lowercased, punctuation and articles removed), 1,000 questions each, 95% Wilson intervals:

| Dataset | Ours (Qwen2.5-7B) | 95% CI | Paper (GPT-4o-mini) | Gap |
|---|---|---|---|---|
| HotpotQA | **54.1%** | 51.0 to 57.2 | 67.20 | 13.1 points |
| 2WikiMultiHopQA | **54.0%** | 50.9 to 57.1 | 70.30 | 16.3 points |
| MuSiQue | **19.1%** | 16.8 to 21.7 | 34.40 | 15.3 points |
| Medical (2,062 questions) | not scored | | 68.40 (LLM-Acc) | needs an LLM judge |

Main observations:

- The ordering of datasets matches the paper: HotpotQA and 2Wiki are similar, MuSiQue is clearly hardest.
- The weak spot is genuine multi-hop questions (section 5.2): for example 2Wiki compositional questions are at 26.6% while comparison questions are at 88.9%.
- Brace-repair of malformed LLM output was needed for a substantial share of schema-extraction replies (section 5.4). Only 17 unique replies could not be repaired and were skipped.
- We cannot attribute the gap to the model alone: we ran no controls (no-retrieval or plain-RAG baseline with the same model). This is a hypothesis, not a measured result.

---

## 2. Setup

| Item | Value |
|---|---|
| Cluster | TU/e course account on Snellius, partition `gpu_mig` (job sees 19.6 GiB GPU memory) |
| Python | module `2023` + `Python/3.11.3-GCCcore-12.3.0`, plain venvs |
| MemGraphRAG venv | upstream `requirements.txt` without `vllm`, plus `gritlm` and `json-repair`; torch 2.5.1+cu124 |
| LLM server | vLLM 0.30.0 in its own venv; model `Qwen/Qwen2.5-7B-Instruct`, served as `qwen2.5-7b-instruct`, bf16 |
| vLLM flags | `--max-model-len 32768 --gpu-memory-utilization 0.85 --enforce-eager --host 127.0.0.1 --port <free port>` and `--override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0}'`; env `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Embedding model | `BAAI/bge-large-en-v1.5` (safetensors) |
| Corpora | `dataset/medical/medical.txt`, `dataset/2wikimutlhopqa/2wikimultihopqa.txt`, `dataset/hotpotqa/hotpotqa.txt`, `dataset/musique/musique.txt` |
| Questions | `dataset/medical/question.json` (2,062), 2Wiki (1,000), HotpotQA (1,000), MuSiQue (1,000, grouped format) |

Commands as in the upstream README (indexing has no ontology flags, so the code's defaults are used):

```
python code/index.py --corpus <corpus.txt> --save-dir <out> --llm-name qwen2.5-7b-instruct \
  --llm-base-url http://127.0.0.1:<port>/v1 --embedding-model <bge path> --tokenizer <bge path> \
  --chunk-size 256 --chunk-overlap 32 --artifact-mode default --memory-max-workers 8

python code/retrieval_dataset_test.py --questions <questions.json> --save-dir <out> --output <qa_results.json> \
  --llm-name qwen2.5-7b-instruct --llm-base-url http://127.0.0.1:<port>/v1 --embedding-model <bge path> \
  --question-type all --sample-num 0 --skip-fact-rerank true --fact-similarity-threshold 0.4 --use-raw-threshold-filter true
```

---

## 3. Deviations from the released code

| # | Deviation | Why | Effect |
|---|---|---|---|
| D1 | LLM is Qwen2.5-7B-Instruct via vLLM, not an OpenAI model | No budget for paid APIs | Lower accuracy than the paper |
| D2 | Patch to `_memory_json`: if the repo's own parsers fail, remove an extra pair of braces (Qwen wraps each item as `{ {...} }`), then try `json_repair`, else log the reply and return `{}` | Qwen-7B returns this malformed shape; without a fix schema extraction crashes | 12,612 unique replies went through the repair path; 17 were skipped (section 5.4) |
| D3 | Guard in conflict resolution: if the parsed reply is empty, return the file's own "unresolved conflict" shape instead of an empty result | An empty dict would otherwise lack `resolved_triples` and could be read as "resolved to nothing" | Only matters for the 17 skipped replies |
| D4 | Conflict resolution treats "maximum context length" errors like the repo's content-filter errors (second attempt with a shorter prompt, or the group is left unresolved) | One MuSiQue conflict group produced a prompt above 31,000 tokens, above Qwen's 32,768 limit | One rejected request in the MuSiQue rerun (section 5.4) |
| D5 | `--memory-max-workers 8` | Speed only | None on results |
| D6 | vLLM sampling forced to temperature 0, no repetition penalty | Qwen's default config (repetition penalty 1.05) made JSON output worse | Closer to the paper's "temperature 0" |

All code changes are in `patches/memgraph_patches_v5.patch` (a `git diff` against the pinned upstream commit). Datasets medical, 2Wiki and HotpotQA ran without D4 (their runs never reached the situation it handles); MuSiQue ran with all patches.

---

## 4. Procedure

1. Check the account and budget: `accinfo`. Compute nodes have internet, so installs and downloads run inside `sbatch` jobs.
2. Create two venvs: one for MemGraphRAG, one for vLLM (separate, because their version pins conflict). Download `bge-large-en-v1.5` and `Qwen2.5-7B-Instruct` into `~/hf_cache` with `HF_HOME`.
3. Clone upstream, `git checkout cd6fabda1ec31a302bb283089e17c087ab7713e8`, apply `patches/memgraph_patches_v5.patch`.
4. Run `scripts/run_repo.sh` for each dataset (one `sbatch` job each). The script: picks a free local port, starts vLLM on `127.0.0.1`, waits for its startup line, sends a test request and aborts unless the job's own vLLM log shows it, then runs the two README commands and prints timing and LLM-call counts.
   ```
   sbatch --time=08:00:00 --job-name=<name> \
     --export=ALL,TAG=<tag>,CORPUS=<corpus>,QUESTIONS=<questions>,SAMPLE_NUM=0 scripts/run_repo.sh
   ```
5. If a job stops, resubmit with the same `TAG`: completed LLM calls are cached in `<out>/llm_cache`.
6. Score: `scripts/score_qa_strict.py` (accuracy with confidence intervals) and `scripts/score_by_strict.py` (by question type).

Smoke test before the full runs: the first 5 HotpotQA questions with their paragraphs (about 28 chunks); then a 100 KB slice of the medical corpus (97 chunks, about 9 minutes). This is how run time and the number of LLM calls per chunk (about 19) were estimated. The files of that first stage (job scripts, logs, tiny test set) are already on `main` under `snellius-work-memgraph/`.

---

## 5. Results

### 5.1 Accuracy

Strict scoring (word-boundary containment). The repository itself computes no accuracy (`retrieval_dataset_test.py` runs QA with `gold_answers=None`), so scoring is our own.

| Dataset | n | Strict | Loose (substring) |
|---|---|---|---|
| HotpotQA | 1,000 | 54.1% | 55.3% |
| 2WikiMultiHopQA | 1,000 | 54.0% | 54.8% |
| MuSiQue | 1,000 | 19.1% | 19.6% |

The loose version counts a gold answer such as `no` as correct when it appears inside another word (for example `unknown`), so it overstates. Strict is the number to report.

`predicted_answer` in `qa_results.json` is already the extracted final answer, so the "full output" and "final answer" columns of the scorer are identical.

### 5.2 By question type (strict)

| Dataset | Type | n | Accuracy |
|---|---|---|---|
| HotpotQA | bridge | 811 | 49.8% |
| HotpotQA | comparison | 189 | 72.5% |
| 2Wiki | comparison | 244 | 88.9% |
| 2Wiki | bridge_comparison | 235 | 76.6% |
| 2Wiki | compositional | 413 | 26.6% |
| 2Wiki | inference | 108 | 30.6% |

HotpotQA's `level` is "hard" for all 1,000 bundled questions, so there is no breakdown by level.

### 5.3 Size, time and cost

| | medical | 2Wiki | HotpotQA | MuSiQue |
|---|---|---|---|---|
| Chunks indexed | 1,030 | 2,787 | 5,472 | 5,659 |
| Schemas / facts / passages in final memory | 8,149 / 13,221 / 1,024 | 14,416 / 53,915 / 2,771 | 35,420 / 101,313 / 5,433 | 41,767 / 103,471 / 5,627 |
| Unique triples before conflict handling | 15,337 | 58,143 | 111,392 | 115,093 |
| Indexing LLM calls (cache rows) | 21,906 | 76,020 | 148,554 | 149,139 |
| Questions | 2,062 | 1,000 | 1,000 | 1,000 |
| Retrieval time | 761 s | 1,178 s | 2,114 s | 2,477 s |
| QA time | 2,705 s | 1,072 s | 927 s | 1,043 s |
| Seconds per question (retrieval + QA) | 1.68 | 2.25 | 3.04 | 3.52 |
| QA prompt / completion tokens | 6.83M / 0.42M | 3.96M / 0.13M | 3.81M / 0.10M | 3.84M / 0.13M |

Indexing wall time is not given: the logs of the completed runs only cover the part left after the cache (for example 21 minutes for medical). The number of LLM calls is the reliable measure of indexing work. Throughput on one `gpu_mig` slice was about 5 LLM calls per second.

Jobs (Elapsed from `sacct`; all on `gpu_mig`):

| Job | Dataset | Elapsed | State | Note |
|---|---|---|---|---|
| 27860023 | medical | 01:33:38 | FAILED | prompt over 8,192-token context in conflict detection |
| 27860579 | 2Wiki | 03:31:05 | FAILED | schema JSON error |
| 27860583 | HotpotQA | 06:34:51 | FAILED | schema JSON error |
| 27860585 | MuSiQue | 06:47:46 | FAILED | schema JSON error |
| 27905882 | medical | 01:20:41 | COMPLETED | |
| 27905883 | 2Wiki | 03:55:04 | COMPLETED | |
| 27905884 | HotpotQA | 07:21:25 | COMPLETED | |
| 27905886 | MuSiQue | 05:51:01 | FAILED | one conflict group over 32,768 tokens |
| 27929039 | MuSiQue | 01:30:27 | COMPLETED | resumed from cache |

These nine jobs total about 38.4 hours, roughly 2,460 SBU at 64 SBU per hour (estimate, not read from `accinfo`). Two earlier attempts that ended after a few minutes with connection errors and the setup and smoke-test jobs are not included.

### 5.4 Reliability of LLM output

| Measure | Value |
|---|---|
| Unique schema replies that needed the repair path (`bad_json.log`) | 12,612 (about 4% of roughly 300,000 schema-extraction calls; calls counted from unique triples) |
| Unique replies skipped as unrepairable (`unrepairable_json.log`) | 17 |
| MuSiQue rerun: requests rejected for context length (`400 Bad Request` in the vLLM log) | 1 |
| Connection errors in the final runs | 0 |

"Unique" means unique reply text: the logs accumulate across reruns and cached replies are logged again, so raw line counts overstate. Whether the one rejected MuSiQue group was resolved by the shorter second prompt or left unresolved was not determined, because the repository does not store the conflict audit in the output folder.

---

## 6. Problems encountered and fixes

| Problem | Cause | Fix |
|---|---|---|
| `ZeroDivisionError` after hours; every LLM request returned "Connection error" | A node is shared by several jobs. vLLM was on fixed port 8000 bound to all addresses, and the readiness check was answered by another job's server | Free random port, bind to 127.0.0.1, require our own vLLM log to show a test request before indexing |
| vLLM crashed at warm-up: `Could not find nvcc` | FlashInfer sampler compiles a kernel at start and compute nodes have no CUDA compiler | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Schema extraction crashed with `Expecting property name enclosed in double quotes: line 4 column 7` | Qwen-7B wraps every item in an extra pair of braces; the first repair attempt (`json_repair` only) sometimes returned a non-dict | Patch D2 (deterministic brace removal first) and D3 |
| Conflict detection or resolution crashed with a 400 error | Prompt larger than the vLLM context limit (8,192, then 32,768 tokens) | Context limit raised to 32,768; patch D4 for the one group still too large |
| Wasted hours on failed runs | Failures appeared after the expensive stages | LLM calls are cached, so reruns with the same tag resume; the job script now stops early when the test request fails |

---

## 7. Limitations

- One run per dataset at temperature 0; confidence intervals reflect question sampling only.
- No control runs, so the effect of the small LLM, the patches and the retrieval stage cannot be separated.
- Accuracy is our own string scorer. The paper's string accuracy is described as containment after lowercasing; exact normalisation may differ. LLM-Acc was not computed, and the medical dataset has no accuracy because its gold answers are long descriptions.
- The number of passages used per question in the QA step was not checked against the paper's top-5 setting.
- The paper's numbers use GPT-4o-mini for indexing, answering and judging. Our numbers should not be compared with them as a pass or fail test.

---

## 8. Files

```
scripts/run_repo.sh              job script (vLLM on a private port, README commands)
scripts/score_qa_strict.py       accuracy with 95% Wilson intervals (strict)
scripts/score_qa.py              loose variant, kept for comparison
scripts/score_by_strict.py       accuracy by question type (strict)
scripts/score_by.py              loose variant
patches/memgraph_patches_v5.patch  all changes to upstream (D2 to D4)
results/                         score outputs and per-run summaries
```

Not included: `qa_results.json` files and graph folders (large), the venvs and model weights.
