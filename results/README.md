# WebQA test results

`webqa-tuned-v1-test-single/` is the raw output of the final WebQA test run (120 test questions,
single worker, beta = 0.25), copied unchanged from
`/scratch-shared/scur0465/mgrv-webqa-full-v1/runs/webqa-tuned-v1-test-single` on Snellius.
It was produced by `scripts/webqa_test_single_lowmem.sbatch` (Slurm job 28514250, 2026-10-23).

`webqa-tuned-v1-test-single/evaluation_report/` is the output of
`evaluation/evaluate_webqa.py` on that folder. Re-running the evaluator on this copy gives the
same scores and input hashes.

`webqa_test_results.tex` holds the paper tables and text, including the paired comparison
with the dense-retrieval baselines in `evaluation_results/webqa_run_{T,C}/recall_test.json`
(on `main`).

## Files not in git

Three parts of the run are too large for git (GitHub's limit is 100 MB per file). They are in the
release asset `webqa-tuned-v1-test-single-large-files.tar.gz` of the GitHub release
`webqa-test-results-v1`:

| Path in the run folder | Size | SHA-256 |
|---|---:|---|
| `diagnostics.jsonl` | 477 MB | `9931d972bace45bc6ba1d672f6b7f72d86bc9f80c1e19d7b07d2f16845de669a` |
| `retrieval_details.jsonl` | 478 MB | `b705a7f8c9c3d0f015a7db9d02c4f0617f3785535f663a56cf451a8f10b13be2` |
| `retrieved/` (120 files) | 530 MB | |

Archive SHA-256: `ed4f466912a26dd85159f3171b034e1f8bcabdaad974cb4b38a9d4442bd5213b (342 MB)`

To restore the complete run folder, extract the archive from this directory:

```bash
gh release download webqa-test-results-v1 -R ULLM-Group-6/memgraphrag-v
tar xzf webqa-tuned-v1-test-single-large-files.tar.gz
```

The evaluator needs only the files in git.

## Extra numbers in the paper

The short-answer match (28.33%), the abstention rate (50%) and the paired comparison with the
dense-retrieval baselines (+26.25 and +8.33 points) come from `evaluation/extra_metrics.py`,
saved in `webqa-tuned-v1-test-single/evaluation_report/extra_metrics.json`. To reproduce, from
the repository root:

```bash
python evaluation/extra_metrics.py results/webqa-tuned-v1-test-single/evaluation_report \
  --questions dataset/test_qa.json \
  --baseline run_T=evaluation_results/webqa_run_T/recall_test.json \
  --baseline run_C=evaluation_results/webqa_run_C/recall_test.json
```

The two `recall_test.json` files are unchanged copies from `main`.
